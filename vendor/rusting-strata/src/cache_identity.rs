//! Metadata identities for model-specific tuning and expert-placement caches.
//!
//! This is deliberately not a content checksum: hashing hundreds of gigabytes
//! of weights on every load would defeat fast startup. Canonical path, file
//! length and modification time distinguish normal local revisions; the caller
//! combines the identity of every shard. Files modified while preserving all
//! these metadata values require explicit cache removal.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::Metadata;
use std::path::Path;
use std::time::UNIX_EPOCH;

fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// Identity of the file whose already-open handle supplied `metadata`.
/// Canonicalization collapses relative/symlink aliases. Paths are hashed in a
/// lossless platform representation, never through a lossy Unicode conversion.
pub fn file_identity(path: &Path, metadata: &Metadata) -> Result<String> {
    let path = path.canonicalize().with_context(|| format!("resolve model cache identity for {}", path.display()))?;
    let mut hash = Sha256::new();
    field(&mut hash, b"forge9070-file-metadata-v1");
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        field(&mut hash, b"unix");
        field(&mut hash, path.as_os_str().as_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        field(&mut hash, b"windows");
        let bytes: Vec<u8> = path.as_os_str().encode_wide().flat_map(u16::to_le_bytes).collect();
        field(&mut hash, &bytes);
    }
    #[cfg(not(any(unix, windows)))]
    {
        field(&mut hash, b"other-platform");
        field(&mut hash, path.as_os_str().as_encoded_bytes());
    }
    field(&mut hash, &metadata.len().to_le_bytes());
    let modified = metadata.modified().context("read model modification time for cache identity")?;
    let (before_epoch, elapsed) = match modified.duration_since(UNIX_EPOCH) {
        Ok(elapsed) => (0u8, elapsed),
        Err(error) => (1u8, error.duration()),
    };
    field(&mut hash, &[before_epoch]);
    field(&mut hash, &elapsed.as_secs().to_le_bytes());
    field(&mut hash, &elapsed.subsec_nanos().to_le_bytes());
    Ok(crate::hf::hex(&hash.finalize()))
}

/// Combine identities in their supplied order, with domain and length framing.
/// The domain keeps a shard list separate from a base/companion combination.
pub fn combine_identities<I, S>(domain: &str, identities: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut hash = Sha256::new();
    field(&mut hash, b"forge9070-cache-identity-v1");
    field(&mut hash, domain.as_bytes());
    for identity in identities { field(&mut hash, identity.as_ref().as_bytes()); }
    crate::hf::hex(&hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, FileTimes};
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, SystemTime};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
            let path = std::env::temp_dir().join(format!("forge-cache-identity-{}-{now}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

    fn write(path: &Path, bytes: &[u8], stamp: u64) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        file.set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(stamp))).unwrap();
    }
    fn id(path: &Path) -> String { file_identity(path, &File::open(path).unwrap().metadata().unwrap()).unwrap() }

    #[test]
    fn identical_names_and_metadata_in_different_repositories_do_not_alias() {
        let temp = Temp::new();
        let (a, b) = (temp.0.join("repo-a/revision/model.gguf"), temp.0.join("repo-b/revision/model.gguf"));
        write(&a, b"same bytes", 1_700_000_000);
        write(&b, b"same bytes", 1_700_000_000);
        assert_eq!(std::fs::metadata(&a).unwrap().modified().unwrap(), std::fs::metadata(&b).unwrap().modified().unwrap());
        assert_ne!(id(&a), id(&b));
    }

    #[test]
    fn a_new_mtime_or_length_invalidates_the_same_path() {
        let temp = Temp::new();
        let path = temp.0.join("model.gguf");
        write(&path, b"aaaa", 1_700_000_000);
        let original = id(&path);
        write(&path, b"bbbb", 1_700_000_001);
        assert_ne!(original, id(&path), "same-sized replacement is identified by its modification time");
        write(&path, b"longer", 1_700_000_000);
        assert_ne!(original, id(&path), "different length invalidates the key even with an unchanged timestamp");
    }

    #[cfg(unix)]
    #[test]
    fn canonical_relative_and_symlink_aliases_share_one_identity() {
        let temp = Temp::new();
        let path = temp.0.join("model.gguf");
        write(&path, b"model", 1_700_000_000);
        let alias = temp.0.join("alias.gguf");
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        assert_eq!(id(&path), id(&alias));
        let mut relative = PathBuf::new();
        for _ in std::env::current_dir().unwrap().components().skip(1) { relative.push(".."); }
        relative.push(path.strip_prefix("/").unwrap());
        assert!(!relative.is_absolute());
        assert_eq!(id(&path), id(&relative));
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_paths_are_not_collapsed_by_lossy_conversion() {
        use std::os::unix::ffi::OsStringExt;
        let temp = Temp::new();
        let a = temp.0.join(std::ffi::OsString::from_vec(b"model-\x80.gguf".to_vec()));
        let b = temp.0.join(std::ffi::OsString::from_vec(b"model-\x81.gguf".to_vec()));
        write(&a, b"same", 1_700_000_000);
        write(&b, b"same", 1_700_000_000);
        assert_eq!(a.to_string_lossy(), b.to_string_lossy());
        assert_ne!(id(&a), id(&b));
    }

    #[test]
    fn combined_keys_preserve_domain_order_and_boundaries() {
        // Preserve the on-disk key format across digest dependency upgrades.
        assert_eq!(combine_identities("shards", ["a", "b"]), "df5d5a499eeb8bbebb093d0f24c3c06fa3a3c44b6acc7b58a65762a82ccf448c");
        assert_ne!(combine_identities("shards", ["a", "bc"]), combine_identities("shards", ["ab", "c"]));
        assert_ne!(combine_identities("shards", ["a", "b"]), combine_identities("shards", ["b", "a"]));
        assert_ne!(combine_identities("shards", ["a", "b"]), combine_identities("companion", ["a", "b"]));
    }

    #[test]
    fn changing_a_later_gguf_shard_invalidates_the_complete_model() {
        let temp = Temp::new();
        let first = temp.0.join("model-00001-of-00002.gguf");
        let second = temp.0.join("model-00002-of-00002.gguf");
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        header.extend(11u64.to_le_bytes());
        header.extend(b"split.count");
        header.extend(2u32.to_le_bytes()); // GGUF uint16 value.
        header.extend(2u16.to_le_bytes());
        header.resize(header.len().next_multiple_of(32), 0);
        write(&first, &header, 1_700_000_000);
        write(&second, &header, 1_700_000_000);
        let first_before = id(&first);
        let before = crate::gguf::Gguf::open(&first).unwrap().identity;
        write(&second, &header, 1_700_000_001);
        assert_eq!(first_before, id(&first));
        let after = crate::gguf::Gguf::open(&first).unwrap().identity;
        assert_ne!(before, after);
    }
}
