use std::path::{Path, PathBuf};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub enum UndoOp {
    Mute { dump_path: PathBuf, start_frame: u64, end_frame: u64 },
    Normalize { dump_path: PathBuf, start_frame: u64, end_frame: u64, dbfs: f64 },
    DeleteSelection { dump_path: PathBuf, start_frame: u64, end_frame: u64, deleted_samples: Vec<f32>, markers_delta: Vec<MarkerDelta> },
}

#[derive(Debug, Clone)]
pub struct MarkerDelta { pub marker_index: usize, pub old_pos_samples: u64, pub new_pos_samples: u64, pub deleted: bool }

#[derive(Debug, Clone)]
pub struct UndoEntry { pub op: UndoOp, pub file_path: PathBuf }

pub struct Stack {
    undo_stack: VecDeque<UndoEntry>, redo_stack: VecDeque<UndoEntry>,
    max_depth: usize, disk_limit_bytes: u64, current_disk_usage: u64, undo_dir: PathBuf,
}

impl Stack {
    pub fn new(undo_dir: PathBuf, max_depth: usize, disk_limit_mb: usize) -> Self {
        let _ = std::fs::create_dir_all(&undo_dir);
        Self { undo_stack: VecDeque::new(), redo_stack: VecDeque::new(), max_depth, disk_limit_bytes: disk_limit_mb as u64 * 1024 * 1024, current_disk_usage: 0, undo_dir }
    }
    pub fn push(&mut self, entry: UndoEntry) {
        self.clear_redo();
        while self.undo_stack.len() >= self.max_depth {
            if let Some(old) = self.undo_stack.pop_front() { self.delete_entry_files(&old); }
        }
        if let Some(size) = Self::dump_size(&entry) {
            self.current_disk_usage += size;
            while self.current_disk_usage > self.disk_limit_bytes && !self.undo_stack.is_empty() {
                if let Some(old) = self.undo_stack.pop_front() {
                    if let Some(size) = Self::dump_size(&old) { self.current_disk_usage -= size; }
                    self.delete_entry_files(&old);
                }
            }
        }
        self.undo_stack.push_back(entry);
    }
    pub fn pop_undo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.undo_stack.pop_back() { self.redo_stack.push_back(entry.clone()); Some(entry) } else { None }
    }
    pub fn pop_redo(&mut self) -> Option<UndoEntry> {
        if let Some(entry) = self.redo_stack.pop_back() { self.undo_stack.push_back(entry.clone()); Some(entry) } else { None }
    }
    pub fn can_undo(&self) -> bool { !self.undo_stack.is_empty() }
    pub fn can_redo(&self) -> bool { !self.redo_stack.is_empty() }
    pub fn clear(&mut self) {
        for entry in self.undo_stack.drain(..) { self.delete_entry_files(&entry); }
        self.clear_redo(); self.current_disk_usage = 0;
    }
    fn clear_redo(&mut self) { for entry in self.redo_stack.drain(..) { self.delete_entry_files(&entry); } }
    fn delete_entry_files(&self, entry: &UndoEntry) {
        match &entry.op {
            UndoOp::Mute { dump_path, .. } | UndoOp::Normalize { dump_path, .. } | UndoOp::DeleteSelection { dump_path, .. } => { let _ = std::fs::remove_file(dump_path); }
        }
    }
    fn dump_size(entry: &UndoEntry) -> Option<u64> {
        let path = match &entry.op {
            UndoOp::Mute { dump_path, .. } | UndoOp::Normalize { dump_path, .. } | UndoOp::DeleteSelection { dump_path, .. } => dump_path,
        };
        std::fs::metadata(path).ok().map(|m| m.len())
    }
    pub fn new_dump_path(&self) -> PathBuf { self.undo_dir.join(format!("undo_{}.raw", uuid::Uuid::new_v4())) }
    pub fn save_dump(&self, path: &Path, samples: &[f32]) -> std::io::Result<()> {
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(path, bytes)
    }
    pub fn load_dump(&self, path: &Path) -> std::io::Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
impl Drop for Stack { fn drop(&mut self) { self.clear(); } }
