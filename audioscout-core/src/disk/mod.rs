use std::path::PathBuf;

/// Информация о диске/точке монтирования
#[derive(Debug, Clone)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: PathBuf,
    pub fs_type: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub is_removable: bool,
}

/// Трейт для получения информации о дисках
pub trait DiskProvider {
    /// Получить список дисков
    fn list_disks(&self) -> Vec<DiskInfo>;
    
    /// Обновить информацию о дисках
    fn refresh(&mut self);
}

/// Провайдер дисков на основе sysinfo
pub struct SysinfoDiskProvider {
    disks: sysinfo::Disks,
}

impl SysinfoDiskProvider {
    pub fn new() -> Self {
        let mut disks = sysinfo::Disks::new();
        disks.refresh(true);
        Self { disks }
    }
    
    /// Проверить, является ли файловая система системной (не показывать)
    fn is_system_fs(fs_type: &str) -> bool {
        matches!(
            fs_type,
            "proc" | "sysfs" | "devtmpfs" | "tmpfs" | "cgroup" | "cgroup2" |
            "securityfs" | "pstore" | "debugfs" | "hugetlbfs" | "mqueue" |
            "binfmt_misc" | "configfs" | "fusectl" | "tracefs" | "bpf" |
            "autofs" | "efivarfs" | "devpts" | "squashfs"
        )
    }
    
    /// Определить имя диска для отображения
    fn disk_name(mount_point: &std::path::Path, device_name: &str) -> String {
        // Если это домашний каталог
        if let Some(home) = dirs_next::home_dir() {
            if mount_point == home {
                return "Home".to_string();
            }
        }
        
        // Если это корень
        if mount_point == std::path::Path::new("/") {
            return "/".to_string();
        }
        
        // Если это в /media/ или /run/media/ или /mnt/
        let mount_str = mount_point.to_string_lossy();
        if mount_str.starts_with("/media/") || mount_str.starts_with("/run/media/") {
            // Извлечь имя диска из пути
            if let Some(name) = mount_point.file_name() {
                return name.to_string_lossy().to_string();
            }
        }
        
        if mount_str.starts_with("/mnt/") {
            if let Some(name) = mount_point.file_name() {
                return name.to_string_lossy().to_string();
            }
        }
        
        // Fallback: использовать имя устройства
        if !device_name.is_empty() {
            device_name.to_string()
        } else {
            mount_point.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| mount_str.to_string())
        }
    }
}

impl DiskProvider for SysinfoDiskProvider {
    fn list_disks(&self) -> Vec<DiskInfo> {
        self.disks.list().iter()
            .filter(|disk| {
                let fs_type = disk.file_system().to_string_lossy();
                !Self::is_system_fs(&fs_type)
            })
            .map(|disk| {
                let mount_point = disk.mount_point().to_path_buf();
                let device_name = disk.name().to_string_lossy().to_string();
                
                DiskInfo {
                    name: Self::disk_name(&mount_point, &device_name),
                    mount_point,
                    fs_type: disk.file_system().to_string_lossy().to_string(),
                    total_bytes: disk.total_space(),
                    available_bytes: disk.available_space(),
                    is_removable: disk.is_removable(),
                }
            })
            .collect()
    }
    
    fn refresh(&mut self) {
        self.disks.refresh(true);
    }
}

impl Default for SysinfoDiskProvider {
    fn default() -> Self {
        Self::new()
    }
}
