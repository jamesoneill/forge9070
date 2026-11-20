//! Offset reads for the owned expert cache. A successful aligned direct read
//! avoids populating the file mapping's OS page cache. Unsupported direct I/O
//! falls back to a reported buffered offset read; it is never described as
//! direct. The only temporary I/O buffer is a bounded 64 KiB aligned allocation.
use anyhow::{Context, Result, ensure};
use std::fs::{File, OpenOptions};
use std::path::Path;

const ALIGNMENT: usize = 4096;
const SCRATCH_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadMethod {
    Direct,
    Buffered,
    #[cfg(test)]
    Memory,
}

pub(super) struct LoadedBytes {
    pub bytes: Vec<u8>,
    pub method: ReadMethod,
    pub direct_fallback: bool,
}

#[derive(Debug)]
pub(super) struct Source {
    buffered: File,
    direct: Option<File>,
    len: u64,
}

impl Source {
    pub(super) fn new(path: &Path, buffered: File, len: u64) -> Self {
        let direct = open_direct(path).filter(|direct| {
            let a = buffered
                .try_clone()
                .ok()
                .and_then(|f| same_file::Handle::from_file(f).ok());
            let b = direct
                .try_clone()
                .ok()
                .and_then(|f| same_file::Handle::from_file(f).ok());
            matches!((a, b), (Some(a), Some(b)) if a == b)
        });
        Self {
            buffered,
            direct,
            len,
        }
    }

    pub(super) fn read(&self, offset: u64, len: usize) -> Result<LoadedBytes> {
        let end = offset
            .checked_add(len as u64)
            .context("expert read range overflow")?;
        ensure!(end <= self.len, "expert read exceeds model shard");
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .context("allocate bounded owned expert bytes")?;
        bytes.resize(len, 0);
        if let Some(file) = self.direct.as_ref() {
            if read_direct(file, offset, &mut bytes).is_ok() {
                return Ok(LoadedBytes {
                    bytes,
                    method: ReadMethod::Direct,
                    direct_fallback: false,
                });
            }
        }
        read_exact_offset(&self.buffered, offset, &mut bytes)
            .context("buffered expert offset read")?;
        Ok(LoadedBytes {
            bytes,
            method: ReadMethod::Buffered,
            direct_fallback: self.direct.is_some(),
        })
    }
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
))]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    // Linux O_DIRECT from the stable userspace open(2) ABI. Other Unix targets
    // deliberately use buffered I/O instead of assuming this flag's value.
    OpenOptions::new()
        .read(true)
        .custom_flags(0o40000)
        .open(path)
        .ok()
}
#[cfg(windows)]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_FLAG_NO_BUFFERING; alignment failures cause a counted fallback.
    OpenOptions::new()
        .read(true)
        .custom_flags(0x2000_0000)
        .open(path)
        .ok()
}
#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")
    ),
    windows
)))]
fn open_direct(_path: &Path) -> Option<File> {
    None
}

fn read_offset(file: &File, offset: u64, output: &mut [u8]) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(output, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(output, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(offset))?;
        file.read(output)
    }
}

fn read_exact_offset(file: &File, mut offset: u64, mut output: &mut [u8]) -> std::io::Result<()> {
    while !output.is_empty() {
        let count = match read_offset(file, offset, output) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            value => value?,
        };
        if count == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        offset += count as u64;
        output = &mut output[count..];
    }
    Ok(())
}

struct Aligned {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}
impl Aligned {
    fn new() -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(SCRATCH_BYTES, ALIGNMENT)?;
        // SAFETY: valid nonzero layout, checked returned pointer, exact layout
        // retained for Drop. Zero initialization makes every exposed byte valid.
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .context("allocate aligned expert I/O scratch")?;
        Ok(Self { pointer, layout })
    }
    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: pointer owns a live SCRATCH_BYTES allocation with no aliases;
        // the slice's borrow is tied to this exclusive borrow of the owner.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), SCRATCH_BYTES) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: this pointer was allocated using the same layout and is
        // exclusively owned by this object; deallocation happens exactly once.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

fn read_direct(file: &File, start: u64, output: &mut [u8]) -> Result<()> {
    if output.is_empty() {
        return Ok(());
    }
    let end = start
        .checked_add(output.len() as u64)
        .context("direct read overflow")?;
    let aligned_end = end
        .checked_add(ALIGNMENT as u64 - 1)
        .context("direct read alignment overflow")?
        / ALIGNMENT as u64
        * ALIGNMENT as u64;
    let mut cursor = start / ALIGNMENT as u64 * ALIGNMENT as u64;
    let mut scratch = Aligned::new()?;
    while cursor < end {
        let requested = (aligned_end - cursor).min(SCRATCH_BYTES as u64) as usize;
        let count = loop {
            match read_offset(file, cursor, &mut scratch.bytes()[..requested]) {
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                value => break value?,
            }
        };
        ensure!(count != 0, "direct expert read reached EOF");
        let copy_start = cursor.max(start);
        let copy_end = (cursor + count as u64).min(end);
        ensure!(
            copy_end > copy_start,
            "direct read did not reach expert payload"
        );
        output[(copy_start - start) as usize..(copy_end - start) as usize].copy_from_slice(
            &scratch.bytes()[(copy_start - cursor) as usize..(copy_end - cursor) as usize],
        );
        if copy_end == end {
            return Ok(());
        }
        ensure!(count == requested, "partial direct read before payload end");
        cursor += count as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    #[test]
    fn unaligned_expert_ranges_and_eof_match_exact_file_bytes() {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("forge-expert-io-{}-{id}", std::process::id()));
        let bytes: Vec<u8> = (0..150_003usize).map(|i| (i * 37 + 11) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let source = Source::new(&path, File::open(&path).unwrap(), bytes.len() as u64);
        for (start, len) in [
            (0, 1),
            (31, 4),
            (4065, 138),
            (17, 100_000),
            (149_999, 4),
            (0, 150_003),
        ] {
            let loaded = source.read(start as u64, len).unwrap();
            assert_eq!(loaded.bytes, &bytes[start..start + len]);
            assert!(matches!(
                loaded.method,
                ReadMethod::Direct | ReadMethod::Buffered
            ));
        }
        assert!(source.read(150_002, 2).is_err());
        let buffered = Source {
            buffered: File::open(&path).unwrap(),
            direct: None,
            len: bytes.len() as u64,
        };
        let loaded = buffered.read(17, 123).unwrap();
        assert_eq!(loaded.method, ReadMethod::Buffered);
        assert!(!loaded.direct_fallback);
        assert_eq!(loaded.bytes, bytes[17..140]);
        let empty_path = path.with_extension("empty");
        std::fs::write(&empty_path, []).unwrap();
        let fallback = Source {
            buffered: File::open(&path).unwrap(),
            direct: Some(File::open(&empty_path).unwrap()),
            len: bytes.len() as u64,
        };
        let loaded = fallback.read(31, 1000).unwrap();
        assert_eq!(loaded.method, ReadMethod::Buffered);
        assert!(loaded.direct_fallback);
        assert_eq!(loaded.bytes, bytes[31..1031]);
        drop(fallback);
        drop(buffered);
        std::fs::remove_file(empty_path).unwrap();
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
}
