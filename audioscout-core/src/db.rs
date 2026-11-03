use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use crate::Marker;

/// Database wrapper for SQLite storage
pub struct Database {
    /// Path of the SQLite file this handle is opened on. Kept so helpers
    /// like [`Database::reopen`] can hand out a *second* connection to the
    /// same database (needed by cross-thread cache purges — see
    /// `preview_worker::delete_previews_for_dir`).
    path: PathBuf,
    conn: Connection,
}

impl Database {
    /// Open or create database at given path
    pub fn open(db_path: &Path) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // Create tables
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS markers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS previews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL,
                mode TEXT NOT NULL,
                uuid TEXT NOT NULL,
                format_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS session (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_markers_path ON markers(path);
            CREATE INDEX IF NOT EXISTS idx_previews_path ON previews(path);
            CREATE INDEX IF NOT EXISTS idx_previews_path_mode ON previews(path, mode);"
        )?;

        Ok(Self { path: db_path.to_path_buf(), conn })
    }

    /// Open a *second, independent* connection to the same database file.
    ///
    /// Needed by cross-thread cache maintenance (e.g. the toolbar "Refresh
    /// preview" purge, which runs on a background thread while the UI thread
    /// still owns its `Database`). SQLite handles concurrent readers/writers
    /// across separate connections safely; the old code instead cloned the
    /// raw handle into the worker thread, which is not a valid SQLite usage
    /// and made those writes invisible/unreliable — one of the reasons the
    /// refresh appeared to delete nothing.
    pub fn reopen(&self) -> Result<Self, rusqlite::Error> {
        Self::open(&self.path)
    }

    /// Save markers for a file
    pub fn save_markers(&self, path: &Path, size: u64, mtime: u64, markers: &[Marker]) -> Result<(), rusqlite::Error> {
        let path_str = path.to_string_lossy().to_string();
        let data = serde_json::to_string(markers).unwrap_or_default();

        // Delete existing
        self.conn.execute(
            "DELETE FROM markers WHERE path = ?1",
            params![path_str],
        )?;

        // Insert new
        self.conn.execute(
            "INSERT INTO markers (path, size, mtime, data) VALUES (?1, ?2, ?3, ?4)",
            params![path_str, size as i64, mtime as i64, data],
        )?;

        Ok(())
    }

    /// Load markers for a file (checks size+mtime match)
    pub fn load_markers(&self, path: &Path, size: u64, mtime: u64) -> Result<Vec<Marker>, rusqlite::Error> {
        let path_str = path.to_string_lossy().to_string();

        let result = self.conn.query_row(
            "SELECT data FROM markers WHERE path = ?1 AND size = ?2 AND mtime = ?3",
            params![path_str, size as i64, mtime as i64],
            |row| row.get::<_, String>(0),
        );

        match result {
            Ok(data) => Ok(serde_json::from_str(&data).unwrap_or_default()),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Save preview cache entry
    pub fn save_preview(&self, path: &Path, size: u64, mtime: u64, mode: &str, uuid: &str) -> Result<(), rusqlite::Error> {
        let path_str = path.to_string_lossy().to_string();

        self.conn.execute(
            "DELETE FROM previews WHERE path = ?1 AND mode = ?2",
            params![path_str, mode],
        )?;

        self.conn.execute(
            "INSERT INTO previews (path, size, mtime, mode, uuid, format_version) VALUES (?1, ?2, ?3, ?4, ?5, 1)",
            params![path_str, size as i64, mtime as i64, mode, uuid],
        )?;

        Ok(())
    }

    /// Load preview UUID for a file
    pub fn load_preview(&self, path: &Path, size: u64, mtime: u64, mode: &str) -> Option<String> {
        let path_str = path.to_string_lossy().to_string();

        self.conn.query_row(
            "SELECT uuid FROM previews WHERE path = ?1 AND size = ?2 AND mtime = ?3 AND mode = ?4 AND format_version = 1",
            params![path_str, size as i64, mtime as i64, mode],
            |row| row.get::<_, String>(0),
        ).ok()
    }

    /// Delete every preview row whose stored path lies directly inside `dir`
    /// (non-recursive). This is the authoritative purge used by the toolbar
    /// "Refresh preview" flow: unlike deletion driven by the *current*
    /// directory listing, it also removes rows for files that have since been
    /// deleted or renamed away from disk, so no orphaned DB entry survives.
    ///
    /// BUGFIX (Refresh preview purged nothing): the old SQL matched rows via a
    /// raw textual prefix (`substr(path,…) = dir || '/'`). Stored paths are
    /// whatever string each writer happened to use — relative (`./music/a.wav`)
    /// or an equivalent-but-different absolute form (e.g. a symlinked session
    /// directory) — none of which share the literal prefix, so the DELETE
    /// matched zero rows and every cache entry survived the refresh. Scope
    /// matching now compares normalised path *components* resolved from the
    /// stored path itself (`Path::parent()`), which stays prefix-safe (siblings
    /// like `/data/music2`, nested subdirectories) while recognising every
    /// spelling of the same location.
    pub fn delete_previews_in_dir(&self, dir: &Path) -> Result<usize, rusqlite::Error> {
        self.delete_all_previews_in_dir(dir)
    }

    /// Snapshot `(path, uuid)` pairs of every cached preview whose **stored
    /// path text** lies directly inside `dir` (non-recursive). The parent
    /// directory is derived from the stored string itself (`Path::parent()`)
    /// and compared component-wise against the normalised `dir`, so rows saved
    /// under any spelling of the same location (`./dir/x`, `dir//x`) are all
    /// returned — with their exact stored text, which is what the deterministic
    /// PNG filename was hashed from. "Refresh preview" uses this list to locate
    /// and unlink the real cache files on disk before deleting the rows.
    pub fn list_previews_in_dir_any_spelling(&self, dir: &Path) -> Result<Vec<(String, String)>, rusqlite::Error> {
        let want: PathBuf = normalize_path(dir);
        if want.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        {
            let mut stmt = self.conn.prepare("SELECT path, uuid FROM previews")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for r in rows.flatten() {
                let parent_ok = match Path::new(&r.0).parent() {
                    Some(parent) => normalize_path(parent) == want,
                    None => false,
                };
                if parent_ok {
                    out.push(r);
                }
            }
        }
        Ok(out)
    }

    /// Delete every preview row whose stored path lies directly inside `dir`,
    /// **including** rows whose PNG filename was derived from a non-normalised
    /// spelling of that path (`./dir/x.wav`, `dir//x.wav`, …).
    ///
    /// This is the authoritative purge used by "Refresh preview": because the
    /// PNG name hashes the *exact* path string, a scope check that only looked
    /// at the normalised text would leave such rows (and their orphaned PNGs)
    /// behind. Here the parent directory is resolved from the stored path
    /// itself (`Path::parent()` — always exact) and compared component-wise to
    /// `dir`, so every spelling maps to the same scope.
    pub fn delete_all_previews_in_dir(&self, dir: &Path) -> Result<usize, rusqlite::Error> {
        let want: PathBuf = normalize_path(dir);
        if want.is_empty() {
            return Ok(0);
        }
        let paths: Vec<String> = {
            let mut stmt = self.conn.prepare("SELECT path FROM previews")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.flatten()
                .filter(|p| match Path::new(p).parent() {
                    Some(parent) => normalize_path(parent) == want,
                    None => false,
                })
                .collect()
        };
        let mut removed = 0usize;
        for path_str in &paths {
            removed += self.conn.execute(
                "DELETE FROM previews WHERE path = ?1",
                params![path_str],
            )?;
        }
        Ok(removed)
    }

    /// Enumerate regular files directly inside `dir` (one level, no
    /// recursion). Unreadable directories yield an empty list.
    pub fn top_level_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().is_file() {
                    out.push(entry.path());
                }
            }
        }
        out
    }

    /// Invalidate previews for a directory
    pub fn invalidate_previews_for_dir(&self, dir: &Path) -> Result<usize, rusqlite::Error> {
        let dir_prefix = format!("{}%", dir.to_string_lossy());
        self.conn.execute(
            "DELETE FROM previews WHERE path LIKE ?1",
            params![dir_prefix],
        )
    }

    /// Delete all previews
    pub fn delete_all_previews(&self) -> Result<usize, rusqlite::Error> {
        self.conn.execute("DELETE FROM previews", [])
    }

    /// Delete all markers
    pub fn delete_all_markers(&self) -> Result<usize, rusqlite::Error> {
        self.conn.execute("DELETE FROM markers", [])
    }

    /// Scan and remove stale markers (file gone or size/mtime mismatch)
    pub fn scan_markers(&self) -> Result<usize, rusqlite::Error> {
        let mut to_delete = Vec::new();

        let mut stmt = self.conn.prepare("SELECT id, path, size, mtime FROM markers")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;

        for row in rows {
            if let Ok((id, path, size, mtime)) = row {
                let path = PathBuf::from(path);
                if !path.exists() {
                    to_delete.push(id);
                    continue;
                }
                if let Ok(meta) = std::fs::metadata(&path) {
                    let file_size = meta.len() as i64;
                    let file_mtime = meta.modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    if file_size != size || file_mtime != mtime {
                        to_delete.push(id);
                    }
                } else {
                    to_delete.push(id);
                }
            }
        }

        let count = to_delete.len();
        for id in to_delete {
            self.conn.execute("DELETE FROM markers WHERE id = ?1", params![id])?;
        }

        Ok(count)
    }

    /// Save session key-value
    pub fn save_session(&self, key: &str, value: &str) -> Result<(), rusqlite::Error> {
        self.conn.execute(
            "INSERT OR REPLACE INTO session (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Load session value
    pub fn load_session(&self, key: &str) -> Option<String> {
        self.conn.query_row(
            "SELECT value FROM session WHERE key = ?1",
            params![key],
            |row| row.get::<_, String>(0),
        ).ok()
    }

    /// Rename path in markers and previews
    pub fn rename_path(&self, old_path: &Path, new_path: &Path) -> Result<(), rusqlite::Error> {
        let old_str = old_path.to_string_lossy().to_string();
        let new_str = new_path.to_string_lossy().to_string();

        self.conn.execute(
            "UPDATE markers SET path = ?1 WHERE path = ?2",
            params![new_str, old_str],
        )?;
        self.conn.execute(
            "UPDATE previews SET path = ?1 WHERE path = ?2",
            params![new_str, old_str],
        )?;

        Ok(())
    }

    /// Get all preview UUIDs from database
    pub fn get_all_preview_uuids(&self) -> Result<Vec<String>, rusqlite::Error> {
        let mut stmt = self.conn.prepare("SELECT uuid FROM previews")?;
        let uuids = stmt.query_map([], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(uuids)
    }
}

/// Normalise a filesystem path for comparison purposes: strip `.` components,
/// collapse repeated separators and drop a trailing separator. Implemented via
/// `Path::components()`, which already performs this on Unix and Windows.
fn normalize_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deletion scoped to a directory must remove only the rows of files
    /// located *directly* inside it: neither sibling directories whose names
    /// share a prefix (`music2`), nor nested subdirectories.
    #[test]
    fn delete_previews_in_dir_is_non_recursive_and_prefix_safe() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("app.db")).unwrap();

        let root = Path::new("/data/music");
        let sibling = Path::new("/data/music2/song.wav");
        let nested = Path::new("/data/music/sub/song.wav");
        let plain = root.join("a.wav");
        let underscored = root.join("b_1.wav");
        let percented = root.join("c%50.wav");

        for p in [sibling, nested, &plain, &underscored, &percented] {
            db.save_preview(p, 10, 20, "wave", &format!("uuid-{}", p.display())).unwrap();
        }

        let removed = db.delete_previews_in_dir(root).unwrap();
        assert_eq!(removed, 3, "only the files directly inside the dir are removed");

        let left: Vec<String> = db.get_all_preview_uuids().unwrap();
        assert_eq!(left.len(), 2);
        assert!(left.contains(&format!("uuid-{}", sibling.display())));
        assert!(left.contains(&format!("uuid-{}", nested.display())));

        // A trailing separator in the queried directory must behave identically.
        db.delete_previews_in_dir(Path::new("/data/music/")).unwrap();
        let left: Vec<String> = db.get_all_preview_uuids().unwrap();
        assert_eq!(left.len(), 2, "trailing slash does not widen/narrow the scope");
    }
}
