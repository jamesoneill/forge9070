//! Positional expert reads from the exact file object used by a GGUF mapping.
//!
//! This is a read-only path. Unbuffered mode is attempted first on Linux and
//! Windows. Alignment/filesystem limitations switch that same open file to a
//! buffered mode, which callers count separately; a pathname is never reopened
//! after a successful open. Mapping headers and sparse PLE rows remains normal
//! OS-managed mmap access regardless of the mode used for expert reads.
//! Read counters describe successful native-flagged versus buffered requests;
//! a filesystem may internally buffer native-flagged reads, so the counters do
//! not prove universal page-cache bypass.

use memmap2::Mmap;
use std::fs::{File, Metadata};
use std::io::{self, ErrorKind};
use std::path::Path;
use std::sync::Mutex;

/// The mode actually used to complete this read, not a memory-residency claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadKind {
    /// A request on an O_DIRECT / FILE_FLAG_NO_BUFFERING handle succeeded.
    /// Count this as a completed native-direct request, not proof that every
    /// filesystem or storage layer bypassed its cache.
    Direct,
    /// Native direct I/O was unavailable or failed; positional buffered read.
    Buffered,
}

// A multiple of common logical and physical sector sizes, including 4 KiB.
// Filesystems with stricter requirements fall back explicitly. One scratch
// chunk caps temporary memory even when a caller requests a large tensor.
const IO_ALIGNMENT: usize = 64 * 1024;
const IO_CHUNK: usize = 1024 * 1024;

struct State {
    file: File,
    direct: bool,
}

pub(super) struct BackingFile {
    state: Mutex<State>,
    len: u64,
}

impl BackingFile {
    pub(super) fn open(path: &Path) -> io::Result<Self> {
        Self::open_mode(path, true)
    }

    fn open_mode(path: &Path, prefer_direct: bool) -> io::Result<Self> {
        let (file, direct) = if prefer_direct {
            match platform::open_direct(path) {
                Ok(file) => (file, true),
                Err(_) => (File::open(path)?, false),
            }
        } else {
            (File::open(path)?, false)
        };
        let len = file.metadata()?.len();
        Ok(Self { state: Mutex::new(State { file, direct }), len })
    }

    fn state(&self) -> io::Result<std::sync::MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| io::Error::other("GGUF backing-file lock poisoned"))
    }

    pub(super) fn metadata(&self) -> io::Result<Metadata> {
        self.state()?.file.metadata()
    }

    pub(super) fn map(&self) -> io::Result<Mmap> {
        let mut state = self.state()?;
        // SAFETY: this is a live read-only File. As with all model mmap access,
        // external writers must not truncate or change a model in use. The OS
        // mapping retains its file-object reference independently of the handle.
        match unsafe { Mmap::map(&state.file) } {
            Ok(map) => Ok(map),
            Err(_) if state.direct => {
                platform::make_buffered(&mut state.file)?;
                state.direct = false;
                unsafe { Mmap::map(&state.file) }
            }
            Err(err) => Err(err),
        }
    }

    pub(super) fn read_region(&self, offset: u64, dst: &mut [u8]) -> io::Result<ReadKind> {
        self.read_region_with(offset, dst, direct_read)
    }

    fn read_region_with(&self, offset: u64, dst: &mut [u8],
        native_read: impl FnOnce(&File, u64, &mut [u8]) -> io::Result<()>) -> io::Result<ReadKind> {
        let end = offset.checked_add(dst.len() as u64)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "GGUF read range overflow"))?;
        if end > self.len {
            return Err(io::Error::new(ErrorKind::UnexpectedEof, "GGUF read lies beyond its original file"));
        }
        let mut state = self.state()?;
        if state.direct {
            match native_read(&state.file, offset, dst) {
                Ok(()) => return Ok(ReadKind::Direct),
                Err(_) => {
                    // Linux clears O_DIRECT on this descriptor; Windows uses
                    // ReOpenFile on this handle. Both retain the same object.
                    platform::make_buffered(&mut state.file)?;
                    state.direct = false;
                }
            }
        }
        read_exact_at(&state.file, offset, dst)?;
        Ok(ReadKind::Buffered)
    }
}

/// Allocate fallibly, then choose an aligned interior slice. No uninitialized
/// bytes or raw allocation ownership cross this boundary.
struct AlignedBuffer {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> io::Result<Self> {
        let total = len.checked_add(IO_ALIGNMENT - 1)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "direct-read scratch size overflow"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(total)
            .map_err(|_| io::Error::new(ErrorKind::OutOfMemory, "cannot allocate aligned direct-read scratch"))?;
        bytes.resize(total, 0);
        let start = (IO_ALIGNMENT - (bytes.as_ptr() as usize % IO_ALIGNMENT)) % IO_ALIGNMENT;
        Ok(Self { bytes, start, len })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }
}

fn direct_read(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    if dst.is_empty() { return Ok(()) }
    let align = IO_ALIGNMENT as u64;
    let end = offset.checked_add(dst.len() as u64)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "direct-read range overflow"))?;
    let mut at = offset / align * align;
    let rounded_end = end.checked_add(align - 1)
        .map(|n| n / align * align)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "direct-read aligned range overflow"))?;
    let scratch_len = (rounded_end - at).min(IO_CHUNK as u64) as usize;
    let mut scratch = AlignedBuffer::new(scratch_len)?;
    while at < end {
        let want = (rounded_end - at).min(scratch_len as u64) as usize;
        let buf = &mut scratch.as_mut()[..want];
        let n = read_retry(file, at, buf)?;
        let got_end = at.checked_add(n as u64)
            .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "direct-read result overflow"))?;
        let copy_start = offset.max(at);
        let copy_end = end.min(got_end);
        if copy_end > copy_start {
            let src = (copy_start - at) as usize;
            let dest = (copy_start - offset) as usize;
            let count = (copy_end - copy_start) as usize;
            dst[dest..dest + count].copy_from_slice(&buf[src..src + count]);
        }
        // A rounded final read may extend beyond EOF. A short read is valid
        // when every requested byte was returned; padding bytes are not data.
        if got_end >= end { return Ok(()) }
        if n == 0 || n % IO_ALIGNMENT != 0 {
            return Err(io::Error::new(ErrorKind::UnexpectedEof, "short unaligned direct GGUF read"));
        }
        at = got_end;
    }
    Ok(())
}

fn read_retry(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
    loop {
        match platform::read_at(file, offset, dst) {
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            result => return result,
        }
    }
}

fn read_exact_at(file: &File, mut offset: u64, mut dst: &mut [u8]) -> io::Result<()> {
    while !dst.is_empty() {
        let n = read_retry(file, offset, dst)?;
        if n == 0 { return Err(io::Error::new(ErrorKind::UnexpectedEof, "short buffered GGUF read")) }
        offset += n as u64;
        dst = &mut dst[n..];
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};

    pub fn open_direct(path: &Path) -> io::Result<File> {
        std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path)
    }
    pub fn make_buffered(file: &mut File) -> io::Result<()> {
        let fd = file.as_raw_fd();
        // SAFETY: fd is a live retained File descriptor. State's mutex excludes
        // concurrent read/flag changes; mmap reads do not use this file status.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags < 0 { return Err(io::Error::last_os_error()) }
        if unsafe { libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_DIRECT) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    pub fn read_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        file.read_at(dst, offset)
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, RawHandle};

    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const GENERIC_READ: u32 = 0x8000_0000;
    const SHARE_READ_WRITE_DELETE: u32 = 0x0000_0007;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn ReOpenFile(original: RawHandle, access: u32, share: u32, flags: u32) -> RawHandle;
    }

    pub fn open_direct(path: &Path) -> io::Result<File> {
        std::fs::OpenOptions::new().read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
    }
    pub fn make_buffered(file: &mut File) -> io::Result<()> {
        // ReOpenFile opens the existing file object, not its possibly replaced
        // pathname. It preserves the object backing the already-created mmap.
        let handle = unsafe { ReOpenFile(file.as_raw_handle(), GENERIC_READ, SHARE_READ_WRITE_DELETE, 0) };
        if handle as isize == -1 || handle.is_null() { return Err(io::Error::last_os_error()) }
        // SAFETY: successful ReOpenFile returned a fresh owned handle.
        *file = unsafe { File::from_raw_handle(handle) };
        Ok(())
    }
    pub fn read_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        file.seek_read(dst, offset)
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod platform {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    pub fn open_direct(_path: &Path) -> io::Result<File> {
        Err(io::Error::new(ErrorKind::Unsupported, "native unbuffered reads unavailable"))
    }
    pub fn make_buffered(_file: &mut File) -> io::Result<()> { Ok(()) }
    pub fn read_at(mut file: &File, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        // The caller's mutex also serializes this portable seek/read fallback.
        file.seek(SeekFrom::Start(offset))?;
        file.read(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    fn fixture() -> (std::path::PathBuf, Vec<u8>) {
        let path = std::env::temp_dir().join(format!("forge-gguf-io-{}-{}", std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed)));
        let bytes: Vec<u8> = (0..IO_CHUNK + IO_ALIGNMENT + 37).map(|i| ((i * 37 + i / 251) % 256) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        (path, bytes)
    }

    #[test]
    fn native_reads_are_exact_for_unaligned_ranges_and_short_eof() {
        let (path, bytes) = fixture();
        let source = BackingFile::open(&path).unwrap();
        for (at, len) in [(0, 1), (13, 997), (IO_ALIGNMENT - 7, 103), (3, IO_CHUNK + 17), (bytes.len() - 101, 101), (bytes.len(), 0)] {
            let mut got = vec![0xcd; len];
            let kind = source.read_region(at as u64, &mut got).unwrap();
            assert!(matches!(kind, ReadKind::Direct | ReadKind::Buffered));
            assert_eq!(got, bytes[at..at + len]);
        }
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn native_path_is_exercised_when_independent_aligned_probe_succeeds() {
        let (path, bytes) = fixture();
        let file = match platform::open_direct(&path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("native_direct_requests=0; capability probe skipped: direct open unavailable: {err}");
                std::fs::remove_file(path).unwrap();
                return;
            }
        };
        let mut probe = AlignedBuffer::new(IO_ALIGNMENT).unwrap();
        let probe_bytes = probe.as_mut();
        assert_eq!(probe_bytes.as_ptr() as usize % IO_ALIGNMENT, 0);
        // Probe the OS primitive directly, before constructing BackingFile.
        // A wrapper regression must not masquerade as a missing capability.
        match read_retry(&file, 0, probe_bytes) {
            Ok(n) if n == IO_ALIGNMENT => assert_eq!(&probe_bytes[..n], &bytes[..n]),
            result => {
                eprintln!("native_direct_requests=0; capability probe skipped: aligned native read unavailable: {result:?}");
                drop(file);
                std::fs::remove_file(path).unwrap();
                return;
            }
        }
        let source = BackingFile {
            len: file.metadata().unwrap().len(),
            state: Mutex::new(State { file, direct: true }),
        };
        let mut got = vec![0xcd; 4096];
        let kind = source.read_region(13, &mut got).unwrap();
        assert_eq!(kind, ReadKind::Direct,
            "aligned OS probe succeeded, but the logical unaligned read fell back");
        assert_eq!(got, bytes[13..13 + 4096]);
        eprintln!("native_direct_requests=1; aligned OS probe and byte-exact unaligned wrapper read succeeded");
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn buffered_fallback_reports_its_mode_and_rejects_out_of_range() {
        let (path, bytes) = fixture();
        let source = BackingFile::open_mode(&path, false).unwrap();
        let mut got = vec![0; 105];
        assert_eq!(source.read_region(19, &mut got).unwrap(), ReadKind::Buffered);
        assert_eq!(got, bytes[19..124]);
        assert!(source.read_region(bytes.len() as u64 - 4, &mut got).is_err());
        assert!(source.read_region(u64::MAX, &mut got).is_err());
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn failed_native_read_retries_buffered_without_trusting_partial_output() {
        let (path, bytes) = fixture();
        let source = BackingFile::open(&path).unwrap();
        let mut got = vec![0; 101];
        let kind = source.read_region_with(37, &mut got, |_file, _offset, dst| {
            dst.fill(0xde);
            Err(io::Error::new(ErrorKind::InvalidInput, "simulated filesystem direct-I/O limitation"))
        }).unwrap();
        assert_eq!(kind, ReadKind::Buffered);
        assert_eq!(got, bytes[37..138]);
        assert_eq!(source.read_region(71, &mut got).unwrap(), ReadKind::Buffered);
        assert_eq!(got, bytes[71..172]);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn replacing_path_does_not_change_retained_expert_source() {
        let (path, bytes) = fixture();
        let source = BackingFile::open(&path).unwrap();
        let map = source.map().unwrap();
        let moved = path.with_extension("old");
        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, vec![0xfe; bytes.len()]).unwrap();
        let mut got = vec![0; 111];
        let kind = source.read_region_with(37, &mut got, |_file, _offset, _dst| {
            Err(io::Error::new(ErrorKind::InvalidInput, "force fallback after pathname replacement"))
        }).unwrap();
        assert_eq!(kind, ReadKind::Buffered);
        assert_eq!(got, bytes[37..148]);
        assert_eq!(map[37..148], bytes[37..148]);
        drop(map);
        drop(source);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(moved).unwrap();
    }
}
