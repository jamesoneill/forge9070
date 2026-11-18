//! Versioned, bounded persistence for derived prompt state. Cache files are
//! optional accelerators: callers discard any record set failing these checks.

use super::{Parked, PartCopy};
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

pub(super) const VERSION: u32 = 2;
pub(super) const MAX_ENTRIES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PartLayout {
    pub slot_bytes: Vec<usize>,
    pub kv_row_bytes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub context: usize,
    pub vocabulary: usize,
    pub parts: Vec<PartLayout>,
}

fn add(a: usize, b: usize) -> Result<usize> { a.checked_add(b).context("prompt cache size overflow") }
fn mul(a: usize, b: usize) -> Result<usize> { a.checked_mul(b).context("prompt cache size overflow") }

impl Layout {
    pub fn record_bytes(&self, tokens: usize) -> Result<usize> {
        ensure!(tokens > 0 && tokens <= self.context, "cached token count is outside the configured context");
        ensure!(self.vocabulary > 0 && !self.parts.is_empty(), "invalid prompt cache layout");
        let mut bytes = mul(tokens, 4)?;
        for p in &self.parts {
            for &n in &p.slot_bytes { bytes = add(bytes, n)?; }
            for &row in &p.kv_row_bytes { bytes = add(bytes, mul(row, tokens)?)?; }
        }
        Ok(bytes)
    }

    /// Maximum serialized size, including the small bounded record headers.
    pub fn max_file_bytes(&self, budget: usize) -> Result<usize> {
        let mut record_header = 16usize; // owner, token count, part count
        for p in &self.parts {
            record_header = add(record_header, 24)?; // PLE history and two counts
            record_header = add(record_header, mul(add(p.slot_bytes.len(), p.kv_row_bytes.len())?, 8)?)?;
        }
        add(add(budget, mul(MAX_ENTRIES, record_header)?)?, 80) // fixed header and checksum
    }

    pub fn validate(&self, pk: &Parked) -> Result<()> {
        let expected = self.record_bytes(pk.ids.len())?;
        ensure!(pk.ids.iter().all(|&id| (id as usize) < self.vocabulary), "cached token outside vocabulary");
        ensure!(pk.parts.len() == self.parts.len(), "prompt cache part count mismatch");
        for (actual, layout) in pk.parts.iter().zip(&self.parts) {
            ensure!(actual.ple_prev.iter().all(|&id| id == -1 || id >= 0 && (id as u64) < self.vocabulary as u64),
                "cached PLE history outside vocabulary");
            ensure!(actual.slot.len() == layout.slot_bytes.len(), "prompt cache slot count mismatch");
            ensure!(actual.kv.len() == layout.kv_row_bytes.len(), "prompt cache KV count mismatch");
            for (buf, &len) in actual.slot.iter().zip(&layout.slot_bytes) {
                ensure!(buf.len() == len, "prompt cache slot length mismatch");
            }
            for (buf, &row) in actual.kv.iter().zip(&layout.kv_row_bytes) {
                ensure!(buf.len() == mul(row, pk.ids.len())?, "prompt cache KV length mismatch");
            }
        }
        ensure!(pk.bytes() == expected, "prompt cache payload size mismatch");
        Ok(())
    }
}

pub(super) fn field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn float_values(hash: &mut Sha256, values: &[f32]) {
    hash.update((values.len() as u64).to_le_bytes());
    let mut bytes = [0u8; 1024];
    for chunk in values.chunks(256) {
        for (dst, value) in bytes.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        hash.update(&bytes[..chunk.len() * 4]);
    }
}

/// Hash actual adapter contents, not just adapter paths or tensor geometry.
/// Adapter order is significant; HashMap iteration order is not.
pub(super) fn adapter_identity(loras: &[crate::model::Lora]) -> [u8; 32] {
    let mut hash = Sha256::new();
    field(&mut hash, b"forge9070-prompt-adapters-v2");
    hash.update((loras.len() as u64).to_le_bytes());
    for lora in loras {
        hash.update(lora.scale.to_bits().to_le_bytes());
        hash.update(lora.alpha.to_bits().to_le_bytes());
        let mut entries: Vec<_> = lora.entries.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        hash.update((entries.len() as u64).to_le_bytes());
        for (name, entry) in entries {
            field(&mut hash, name.as_bytes());
            for n in [entry.rank, entry.in_dim, entry.out_dim, entry.n_expert] { hash.update((n as u64).to_le_bytes()); }
            for matrices in [&entry.a, &entry.b] {
                hash.update((matrices.len() as u64).to_le_bytes());
                for row in matrices { float_values(&mut hash, row); }
            }
        }
    }
    hash.finalize().into()
}

/// GGUF identity is the established shard path/size/mtime identity, not a scan
/// of all weight bytes. Immutable pinned checkpoints are the intended input.
pub(super) fn model_identity(m: &crate::model::Model<'_>) -> [u8; 32] {
    let mut hash = Sha256::new();
    field(&mut hash, b"forge9070-prompt-model-v2");
    field(&mut hash, m.w.g.identity.as_bytes());
    field(&mut hash, m.w.aux_g.map_or("", |g| g.identity.as_str()).as_bytes());
    field(&mut hash, format!("{:?}", m.cfg).as_bytes());
    field(&mut hash, &adapter_identity(&m.loras));
    let mut baked: Vec<_> = m.w.lora_baked.iter().collect();
    baked.sort();
    for name in baked { field(&mut hash, name.as_bytes()); }
    hash.finalize().into()
}

struct HashReader<R> { inner: R, hash: Sha256 }
impl<R: Read> Read for HashReader<R> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
}
struct HashWriter<W> { inner: W, hash: Sha256 }
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

fn u32_value(r: &mut impl Read) -> Result<u32> { let mut b = [0; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn u64_value(r: &mut impl Read) -> Result<u64> { let mut b = [0; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn blob(r: &mut impl Read, expected: usize) -> Result<Vec<u8>> {
    let len = usize::try_from(u64_value(r)?).context("prompt cache buffer length exceeds platform range")?;
    ensure!(len == expected, "prompt cache buffer length {len} != expected {expected}");
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).context("cannot allocate prompt cache buffer")?;
    bytes.resize(len, 0);
    r.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub(super) fn read(reader: impl Read, identity: [u8; 32], layout: &Layout, budget: usize) -> Result<Vec<Parked>> {
    let mut r = HashReader { inner: reader, hash: Sha256::new() };
    let mut magic = [0; 8]; r.read_exact(&mut magic)?;
    ensure!(&magic == b"STRATAPC", "invalid prompt cache magic");
    ensure!(u32_value(&mut r)? == VERSION, "unsupported prompt cache version; old derived state discarded");
    let mut actual = [0; 32]; r.read_exact(&mut actual)?;
    ensure!(actual == identity, "prompt cache checkpoint, adapter or runtime identity mismatch");
    let count = u32_value(&mut r)? as usize;
    ensure!(count <= MAX_ENTRIES, "prompt cache has too many entries");
    let mut loaded = Vec::with_capacity(count);
    let mut bytes = 0usize;
    for _ in 0..count {
        let owner = u64_value(&mut r)?;
        let tokens = u32_value(&mut r)? as usize;
        // Calculate and enforce all payload memory before allocating even IDs.
        bytes = add(bytes, layout.record_bytes(tokens)?)?;
        ensure!(bytes <= budget, "prompt cache payload exceeds available cache RAM");
        let mut ids = Vec::new();
        ids.try_reserve_exact(tokens).context("cannot allocate cached token IDs")?;
        for _ in 0..tokens {
            let id = u32_value(&mut r)?;
            ensure!((id as usize) < layout.vocabulary, "cached token outside vocabulary");
            ids.push(id);
        }
        ensure!(u32_value(&mut r)? as usize == layout.parts.len(), "prompt cache part count mismatch");
        let mut parts = Vec::with_capacity(layout.parts.len());
        for part in &layout.parts {
            let ple_prev = [u64_value(&mut r)? as i64, u64_value(&mut r)? as i64];
            ensure!(ple_prev.iter().all(|&id| id == -1 || id >= 0 && (id as u64) < layout.vocabulary as u64),
                "cached PLE history outside vocabulary");
            ensure!(u32_value(&mut r)? as usize == part.slot_bytes.len(), "prompt cache slot count mismatch");
            let mut slot = Vec::with_capacity(part.slot_bytes.len());
            for &len in &part.slot_bytes { slot.push(blob(&mut r, len)?); }
            ensure!(u32_value(&mut r)? as usize == part.kv_row_bytes.len(), "prompt cache KV count mismatch");
            let mut kv = Vec::with_capacity(part.kv_row_bytes.len());
            for &row in &part.kv_row_bytes { kv.push(blob(&mut r, mul(row, tokens)?)?); }
            let pk_part = PartCopy { ple_prev, slot, kv };
            parts.push(pk_part);
        }
        let pk = Parked { ids, owner, parts };
        layout.validate(&pk)?;
        loaded.push(pk);
    }
    let expected: [u8; 32] = r.hash.finalize().into();
    let mut digest = [0; 32]; r.inner.read_exact(&mut digest)?;
    ensure!(digest == expected, "prompt cache checksum mismatch");
    ensure!(r.inner.read(&mut [0u8; 1])? == 0, "prompt cache has trailing bytes");
    Ok(loaded)
}

pub(super) fn write(writer: impl Write, identity: [u8; 32], layout: &Layout,
    entries: &[&Parked], budget: usize) -> Result<()> {
    ensure!(entries.len() <= MAX_ENTRIES, "too many prompt cache entries to save");
    let mut bytes = 0;
    for pk in entries {
        layout.validate(pk)?;
        bytes = add(bytes, pk.bytes())?;
    }
    ensure!(bytes <= budget, "prompt cache exceeds configured RAM budget");
    let mut w = HashWriter { inner: writer, hash: Sha256::new() };
    w.write_all(b"STRATAPC")?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&identity)?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;
    for pk in entries {
        w.write_all(&pk.owner.to_le_bytes())?;
        w.write_all(&(pk.ids.len() as u32).to_le_bytes())?;
        for id in &pk.ids { w.write_all(&id.to_le_bytes())?; }
        w.write_all(&(pk.parts.len() as u32).to_le_bytes())?;
        for part in &pk.parts {
            for id in part.ple_prev { w.write_all(&id.to_le_bytes())?; }
            for buffers in [&part.slot, &part.kv] {
                w.write_all(&(buffers.len() as u32).to_le_bytes())?;
                for b in buffers { w.write_all(&(b.len() as u64).to_le_bytes())?; w.write_all(b)?; }
            }
        }
    }
    let digest = w.hash.finalize();
    w.inner.write_all(&digest)?;
    w.inner.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Layout, Parked) {
        let layout = Layout { context: 32, vocabulary: 100,
            parts: vec![PartLayout { slot_bytes: vec![8, 4], kv_row_bytes: vec![4, 4] }] };
        let pk = Parked { ids: vec![1, 2, 3], owner: 17,
            parts: vec![PartCopy { ple_prev: [2, 3], slot: vec![vec![4; 8], vec![5; 4]], kv: vec![vec![6; 12], vec![7; 12]] }] };
        (layout, pk)
    }
    fn encoded(layout: &Layout, pk: &Parked) -> Vec<u8> {
        let mut bytes = Vec::new();
        write(&mut bytes, [9; 32], layout, &[pk], 4096).unwrap();
        bytes
    }

    #[test]
    fn roundtrip_preserves_exact_payload_and_owner() {
        let (layout, pk) = fixture();
        let bytes = encoded(&layout, &pk);
        let loaded = read(bytes.as_slice(), [9; 32], &layout, pk.bytes()).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].ids, pk.ids);
        assert_eq!(loaded[0].owner, pk.owner);
        assert_eq!(loaded[0].parts[0].ple_prev, pk.parts[0].ple_prev);
        assert_eq!(loaded[0].parts[0].slot, pk.parts[0].slot);
        assert_eq!(loaded[0].parts[0].kv, pk.parts[0].kv);
    }

    #[test]
    fn rejects_old_version_identity_corruption_and_truncation() {
        let (layout, pk) = fixture();
        let bytes = encoded(&layout, &pk);
        assert!(read(bytes.as_slice(), [8; 32], &layout, 4096).is_err());
        let mut old = bytes.clone(); old[8..12].copy_from_slice(&1u32.to_le_bytes());
        assert!(read(old.as_slice(), [9; 32], &layout, 4096).is_err());
        let mut corrupt = bytes.clone(); let at = corrupt.len() - 33; corrupt[at] ^= 1;
        assert!(read(corrupt.as_slice(), [9; 32], &layout, 4096).is_err());
        assert!(read(&bytes[..bytes.len() - 1], [9; 32], &layout, 4096).is_err());
        let mut trailing = bytes.clone(); trailing.push(0);
        assert!(read(trailing.as_slice(), [9; 32], &layout, 4096).is_err());
    }

    #[test]
    fn rejects_untrusted_counts_and_lengths_before_allocating_payload() {
        let (layout, pk) = fixture();
        let bytes = encoded(&layout, &pk);
        for (offset, field) in [(44, u32::MAX), (56, u32::MAX), (72, u32::MAX), (92, u32::MAX)] {
            let mut bad = bytes.clone(); bad[offset..offset + 4].copy_from_slice(&field.to_le_bytes());
            assert!(read(bad.as_slice(), [9; 32], &layout, 4096).is_err());
        }
        let mut bad = bytes.clone(); bad[96..104].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(read(bad.as_slice(), [9; 32], &layout, 4096).is_err());
        assert!(read(bytes.as_slice(), [9; 32], &layout, pk.bytes() - 1).is_err());
        let mut short = layout.clone(); short.parts[0].kv_row_bytes[0] = 2;
        assert!(read(bytes.as_slice(), [9; 32], &short, 4096).is_err());
    }

    #[test]
    fn rejects_wrong_buffer_geometry_and_invalid_tokens() {
        let (layout, mut pk) = fixture();
        pk.parts[0].slot.pop();
        assert!(layout.validate(&pk).is_err());
        let (_, mut pk) = fixture(); pk.parts[0].kv[0].pop();
        assert!(layout.validate(&pk).is_err());
        let (_, mut pk) = fixture(); pk.ids[0] = 100;
        assert!(layout.validate(&pk).is_err());
        let (_, mut pk) = fixture(); pk.parts[0].ple_prev[0] = -2;
        assert!(layout.validate(&pk).is_err());
    }

    #[test]
    fn adapter_fingerprint_tracks_values_and_scales() {
        use crate::model::{Lora, LoraEntry};
        let entry = LoraEntry { rank: 1, in_dim: 2, out_dim: 2, n_expert: 1,
            a: vec![vec![1., 2.]], b: vec![vec![3., 4.]] };
        let original = Lora { scale: 1., alpha: 2., entries: [("a".into(), entry.clone()), ("b".into(), entry)].into() };
        let expected = adapter_identity(&[original.clone()]);
        let mut reordered = original.clone();
        let mut entries: Vec<_> = original.entries.into_iter().collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0));
        reordered.entries = entries.into_iter().collect();
        assert_eq!(adapter_identity(&[reordered.clone()]), expected);
        reordered.scale = 0.5;
        assert_ne!(adapter_identity(&[reordered.clone()]), expected);
        reordered.scale = 1.;
        reordered.entries.get_mut("a").unwrap().a[0][0] = 9.;
        assert_ne!(adapter_identity(&[reordered]), expected);
        assert_ne!(adapter_identity(&[]), expected);
    }
}
