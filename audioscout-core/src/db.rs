use rusqlite::{Connection, params};
use std::path::{Path, PathBuf};
use crate::Marker;

/// Escape the LIKE wildcards (`%`, `_`) and the escape char itself in a
/// literal prefix so it can be used safely in `LIKE '<prefix>%' ESCAPE '\'`.
fn escape_like_prefix(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            'x' | '%' | '_' => {
                out.push('x');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Database wrapper for SQLite storage
pub struct Database {
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

        Ok(Self { conn })
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

    /// Delete preview rows whose path is directly inside `dir` (non-recursive).
    /// The trailing separator in the prefix keeps `.../music` from matching
    /// `.../music2/...` and excludes subdirectory entries.
    pub fn delete_previews_in_dir(&self, dir: &Path) -> Result<usize, rusqlite::Error> {
        // Escaped directory + escaped separator are literal; only the final
        // `%` stays a wildcard. `ESCAPE 'x'` avoids the backslash-in-string
        // pitfall (SQLite requires the escape clause to be one character).
        let sep = std::path::MAIN_SEPARATOR.to_string();
        let pattern = format!("{}{}%", escape_like_prefix(&dir.to_string_lossy()), escape_like_prefix(&sep));
        self.conn.execute(
            "DELETE FROM previews WHERE path LIKE ?1 ESCAPE 'x'",
            params![pattern],
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `delete_previews_in_dir` must remove only the rows of files directly
    /// inside the target directory: neither sibling directories whose names
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
        assert_eq!(removed, 3, "only direct children (incl. _ and % in names) match");

        let left: Vec<String> = db.get_all_preview_uuids().unwrap();
        assert_eq!(left.len(), 2);
        assert!(left.contains(&format!("uuid-{}", sibling.display())));
        assert!(left.contains(&format!("uuid-{}", nested.display())));
    }
}
