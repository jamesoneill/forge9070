//! Offset reads for the owned expert cache. A successful aligned direct read
//! avoids populating the file mapping's OS page cache. Unsupported direct I/O
//! falls back to a reported buffered offset read; it is never described as
//! direct. Temporary aligned scratch is at most 1 MiB per active read, and only
//! the aligned span needed for that request is allocated.
use anyhow::{Context, Result, ensure};
use std::fs::{File, OpenOptions};
use std::path::Path;

const ALIGNMENT: usize = 4096;
pub(super) const DEFAULT_SCRATCH_BYTES: usize = 1024 * 1024;
const SCRATCH_ENV: &str = "NEXTORA_EXPERT_READ_KIB";

fn configured_scratch_bytes(value: Option<&std::ffi::OsStr>) -> Result<usize> {
    let Some(value) = value else { return Ok(DEFAULT_SCRATCH_BYTES) };
    let kib: usize = value.to_str().context("NEXTORA_EXPERT_READ_KIB must be a UTF-8 integer")?
        .parse().context("NEXTORA_EXPERT_READ_KIB must be an integer number of KiB")?;
    ensure!((ALIGNMENT / 1024..=DEFAULT_SCRATCH_BYTES / 1024).contains(&kib)
        && kib % (ALIGNMENT / 1024) == 0,
        "NEXTORA_EXPERT_READ_KIB must be 4..1024 KiB in 4 KiB multiples");
    kib.checked_mul(1024).context("expert read scratch size overflow")
}

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
    scratch_bytes: usize,
}

impl Source {
    pub(super) fn new(path: &Path, buffered: File, len: u64) -> Result<Self> {
        let scratch_bytes = configured_scratch_bytes(std::env::var_os(SCRATCH_ENV).as_deref())?;
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
        Ok(Self {
            buffered,
            direct,
            len,
            scratch_bytes,
        })
    }

    pub(super) fn scratch_max_bytes(&self) -> usize { self.scratch_bytes }

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
            if read_direct(file, offset, &mut bytes, self.scratch_bytes).is_ok() {
                return Ok(LoadedBytes {
                    bytes,
                    method: ReadMethod::Direct,
                    direct_fallback: false,
                });
            }
        }
        read_exact_offset(&self.buffered, offset, &mut bytes)
            .context("buffered expert offset read")?;
        #[cfg(target_os = "linux")]
        advise_consumed_buffered_pages(&self.buffered, offset, end);
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

// Only whole pages entirely inside the owned read are eligible. A partial
// page may contain unrelated tensors; never request eviction of that page.
#[cfg(target_os = "linux")]
fn interior_page_range(start: u64, end: u64, page: u64) -> Option<(u64, u64)> {
    if page == 0 || end <= start {
        return None;
    }
    let first = start.checked_add((page - start % page) % page)?;
    let last = end - end % page;
    (last > first).then(|| (first, last - first))
}

#[cfg(target_os = "linux")]
fn advise_consumed_buffered_pages(file: &File, start: u64, end: u64) {
    use std::os::fd::AsRawFd;
    // Experimental: the file cache is shared with mmap readers and other
    // processes. Default off; DONTNEED is advisory, not a memory guarantee.
    if std::env::var_os("NEXTORA_EXPERT_DROP_BUFFERED_CACHE").as_deref()
        != Some(std::ffi::OsStr::new("1"))
    {
        return;
    }
    // SAFETY: sysconf accepts this constant and does not dereference pointers.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let Ok(page) = u64::try_from(page) else { return };
    let Some((first, length)) = interior_page_range(start, end, page) else { return };
    let (Ok(first), Ok(length)) = (libc::off_t::try_from(first), libc::off_t::try_from(length))
        else { return };
    // SAFETY: the live descriptor is borrowed for this call; the range is
    // checked and nonzero. The result is deliberately advisory, not an I/O error.
    let _ = unsafe {
        libc::posix_fadvise(file.as_raw_fd(), first, length, libc::POSIX_FADV_DONTNEED)
    };
}

#[cfg(all(test, target_os = "linux"))]
#[test]
fn cache_advice_never_includes_partial_pages() {
    assert_eq!(interior_page_range(0, 8192, 4096), Some((0, 8192)));
    assert_eq!(interior_page_range(1, 8193, 4096), Some((4096, 4096)));
    assert_eq!(interior_page_range(1, 4096, 4096), None);
    assert_eq!(interior_page_range(0, 4095, 4096), None);
    assert_eq!(interior_page_range(0, 0, 4096), None);
    assert_eq!(interior_page_range(u64::MAX - 1, u64::MAX, 4096), None);
    assert_eq!(interior_page_range(1, 8192, 0), None);
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
    fn new(bytes: usize) -> Result<Self> {
        ensure!(bytes > 0 && bytes <= DEFAULT_SCRATCH_BYTES && bytes % ALIGNMENT == 0,
            "aligned expert scratch is outside its configured bounds");
        let layout = std::alloc::Layout::from_size_align(bytes, ALIGNMENT)?;
        // SAFETY: valid nonzero layout, checked returned pointer, exact layout
        // retained for Drop. Zero initialization makes every exposed byte valid.
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .context("allocate aligned expert I/O scratch")?;
        Ok(Self { pointer, layout })
    }
    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: pointer owns a live layout-sized allocation with no aliases;
        // the slice's borrow is tied to this exclusive borrow of the owner.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: this pointer was allocated using the same layout and is
        // exclusively owned by this object; deallocation happens exactly once.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

fn read_direct(file: &File, start: u64, output: &mut [u8], scratch_limit: usize) -> Result<()> {
    read_direct_with(start, output, scratch_limit, |offset, bytes| read_offset(file, offset, bytes))
}

fn read_direct_with(start: u64, output: &mut [u8], scratch_limit: usize,
    mut read: impl FnMut(u64, &mut [u8]) -> std::io::Result<usize>) -> Result<()> {
    ensure!(scratch_limit > 0 && scratch_limit <= DEFAULT_SCRATCH_BYTES && scratch_limit % ALIGNMENT == 0,
        "direct expert scratch limit must be bounded and alignment-sized");
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
    let aligned_span = aligned_end - cursor;
    let scratch_bytes = usize::try_from(aligned_span.min(scratch_limit as u64))?;
    let mut scratch = Aligned::new(scratch_bytes)?;
    while cursor < end {
        let requested = (aligned_end - cursor).min(scratch_bytes as u64) as usize;
        let count = loop {
            match read(cursor, &mut scratch.bytes()[..requested]) {
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
        let source = Source::new(&path, File::open(&path).unwrap(), bytes.len() as u64).unwrap();
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
            scratch_bytes: DEFAULT_SCRATCH_BYTES,
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
            scratch_bytes: DEFAULT_SCRATCH_BYTES,
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
    #[test]
    fn configured_scratch_is_bounded_aligned_and_supports_controlled_ab_sizes() {
        use std::ffi::OsStr;
        assert_eq!(configured_scratch_bytes(None).unwrap(), 1024*1024);
        for kib in [4usize,64,128,1024] {
            assert_eq!(configured_scratch_bytes(Some(OsStr::new(&kib.to_string()))).unwrap(),kib*1024);
        }
        for invalid in ["", "0", "1", "3", "5", "1025", "2048", "-1", "1MiB", "184467440737095516150"] {
            assert!(configured_scratch_bytes(Some(OsStr::new(invalid))).is_err(),"accepted {invalid}");
        }
    }

    #[test]
    fn direct_reads_copy_exact_bytes_across_scratch_boundaries_with_bounded_allocations() {
        let input:Vec<u8>=(0..2*DEFAULT_SCRATCH_BYTES+123).map(|i|(i*37+11)as u8).collect();
        for limit in [64*1024,DEFAULT_SCRATCH_BYTES] {
            for (start,len) in [(0,1),(4095,2),(17,100000),(DEFAULT_SCRATCH_BYTES-19,8251),
                (17,2*DEFAULT_SCRATCH_BYTES+100),(input.len()-7,7),(0,input.len())] {
                let mut output=vec![0;len];let mut largest=0;let mut calls=0;
                read_direct_with(start as u64,&mut output,limit,|offset,bytes| {
                    calls+=1;largest=largest.max(bytes.len());
                    assert_eq!(offset%ALIGNMENT as u64,0);
                    assert_eq!(bytes.len()%ALIGNMENT,0);
                    assert_eq!(bytes.as_ptr()as usize%ALIGNMENT,0);
                    let offset=offset as usize;
                    let count=bytes.len().min(input.len().saturating_sub(offset));
                    bytes[..count].copy_from_slice(&input[offset..offset+count]);Ok(count)
                }).unwrap();
                assert_eq!(output,input[start..start+len]);
                let span=(start+len).next_multiple_of(ALIGNMENT)-start/ALIGNMENT*ALIGNMENT;
                assert_eq!(largest,span.min(limit));
                assert_eq!(calls,span.div_ceil(limit));
            }
        }
    }

    #[test]
    fn one_mib_batches_an_expert_read_without_changing_bytes() {
        let input:Vec<u8>=(0..DEFAULT_SCRATCH_BYTES).map(|i|(i*13+7)as u8).collect();
        let mut counts=Vec::new();
        for limit in [64*1024,DEFAULT_SCRATCH_BYTES] {
            let mut output=vec![0;884736];let mut calls=0;
            read_direct_with(31,&mut output,limit,|offset,bytes| {
                calls+=1;let offset=offset as usize;
                bytes.copy_from_slice(&input[offset..offset+bytes.len()]);Ok(bytes.len())
            }).unwrap();
            assert_eq!(output,input[31..31+884736]);counts.push(calls);
        }
        assert_eq!(counts,[14,1]);
    }

    #[test]
    fn interrupted_empty_partial_and_overflow_reads_are_handled() {
        let mut output=vec![0;8192];let mut calls=0;
        read_direct_with(0,&mut output,DEFAULT_SCRATCH_BYTES,|_,bytes| {
            calls+=1;
            if calls==1 {return Err(std::io::ErrorKind::Interrupted.into())}
            bytes.fill(17);Ok(bytes.len())
        }).unwrap();
        assert_eq!(calls,2);assert!(output.iter().all(|byte|*byte==17));
        read_direct_with(0,&mut [],DEFAULT_SCRATCH_BYTES,|_,_|panic!("empty read performed I/O")).unwrap();
        assert!(read_direct_with(0,&mut output,DEFAULT_SCRATCH_BYTES,|_,bytes|Ok(bytes.len()/2)).is_err());
        assert!(read_direct_with(0,&mut output,DEFAULT_SCRATCH_BYTES,|_,_|Ok(0)).is_err());
        assert!(read_direct_with(u64::MAX,&mut output,DEFAULT_SCRATCH_BYTES,|_,_|panic!()).is_err());
        assert!(read_direct_with(0,&mut output,0,|_,_|panic!()).is_err());
        assert!(read_direct_with(0,&mut output,DEFAULT_SCRATCH_BYTES+ALIGNMENT,|_,_|panic!()).is_err());
    }

}
