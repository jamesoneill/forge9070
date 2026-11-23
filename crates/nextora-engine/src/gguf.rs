//! Bounded independent reader for complete little-endian GGUF v2/v3 shard sets.
//!
//! Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//! This is a fresh implementation against the format, not a wrapper around an
//! inference library. Immutable mapped weight storage stays in its original
//! format; tensor views borrow their owning mapping and never extend lifetimes.

use crate::quant::{self, QuantType};
use anyhow::{Context, Result, bail, ensure};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "expert_cache.rs"]
mod expert_cache;
#[path = "expert_io.rs"]
mod expert_io;
pub use expert_cache::ExpertCacheStats;
use expert_cache::{ExpertCache, OwnedBytes};

static MODEL_ID: AtomicU64 = AtomicU64::new(1);
const MAX_SHARDS: usize = 1024;

const MAX_ENTRIES: usize = 200_000;
const MAX_ARRAY_ITEMS: usize = 2_000_000;
const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 256 * 1024 * 1024;
const MAX_METADATA_ALLOCATIONS: usize = 128 * 1024 * 1024;
const MAX_ARRAY_DEPTH: usize = 8;
const MAX_ALIGNMENT: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some((*v).into()),
            Self::U16(v) => Some((*v).into()),
            Self::U32(v) => Some((*v).into()),
            Self::U64(v) => Some(*v),
            Self::I8(v) => u64::try_from(*v).ok(),
            Self::I16(v) => u64::try_from(*v).ok(),
            Self::I32(v) => u64::try_from(*v).ok(),
            Self::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        let number = match self {
            Self::F32(v) => *v,
            Self::F64(v) => {
                let narrowed = *v as f32;
                if !v.is_finite() || (*v != 0.0 && narrowed == 0.0) {
                    return None;
                }
                narrowed
            }
            Self::U8(v) => *v as f32,
            Self::U16(v) => *v as f32,
            Self::U32(v) => *v as f32,
            Self::U64(v) => *v as f32,
            Self::I8(v) => *v as f32,
            Self::I16(v) => *v as f32,
            Self::I32(v) => *v as f32,
            Self::I64(v) => *v as f32,
            _ => return None,
        };
        number.is_finite().then_some(number)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        self.as_array()
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: the contiguous row width first, followed by rows and experts.
    pub dims: Vec<u64>,
    pub ty: QuantType,
    /// Byte offset relative to the aligned tensor-data section.
    pub offset: u64,
    pub byte_len: usize,
    pub file_offset: usize,
    /// Index into this model's validated shard set.
    pub shard_index: usize,
}

/// Stable identity for immutable bytes. It never depends on an allocation address
/// and remains unchanged when an expert is copied into or evicted from a cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TensorKey {
    pub model: u64,
    pub shard: u32,
    pub offset: u64,
    pub length: u64,
    pub quantization: u32,
}

pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub alignment: usize,
    pub data_offset: usize,
    mappings: Vec<Arc<Mmap>>,
    sources: Vec<expert_io::Source>,
    paths: Vec<PathBuf>,
    identity: u64,
    cache: ExpertCache,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("tensors", &self.tensors.len())
            .field("mapped_bytes", &self.mapped_bytes())
            .field("shards", &self.paths.len())
            .finish()
    }
}

impl Gguf {
    /// Map the complete model, resolving a standard `-00001-of-00002.gguf`
    /// shard set automatically. Files must remain unchanged while mapped.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_cache(path, 0)
    }

    /// Map weights and optionally retain routed expert byte ranges in a bounded
    /// owned host cache. Capacity counts encoded weight payload, not mmap/page
    /// cache, metadata or GPU allocations. Zero disables the owned cache.
    pub fn open_with_cache(path: impl AsRef<Path>, capacity_bytes: usize) -> Result<Self> {
        let path = path.as_ref();
        let (mapping, parsed, source) = map_file(path)?;
        let count = split_integer(&parsed.meta, "split.count", 1)?;
        let selected = split_integer(&parsed.meta, "split.no", 0)?;
        let paths = shard_paths(path, count, selected)?;
        let identity = MODEL_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("model identity counter exhausted"))?;
        let mut initial = Some((mapping, parsed, source));
        let mut sources = Vec::with_capacity(count);
        let mut mappings = Vec::with_capacity(count);
        let mut tensors = BTreeMap::new();
        let mut first: Option<Parsed> = None;
        let mut total_mapped = 0usize;
        for (index, shard_path) in paths.iter().enumerate() {
            let (map, mut part, source) = if index == selected {
                initial.take().expect("selected shard consumed once")
            } else {
                map_file(shard_path)?
            };
            ensure!(
                split_integer(&part.meta, "split.count", 1)? == count,
                "shard count disagrees: {}",
                shard_path.display()
            );
            ensure!(
                split_integer(&part.meta, "split.no", 0)? == index,
                "shard ordinal disagrees: {}",
                shard_path.display()
            );
            total_mapped = total_mapped
                .checked_add(map.len())
                .context("combined model mapping size overflow")?;
            ensure!(
                total_mapped <= isize::MAX as usize,
                "combined model exceeds addressable mapping size"
            );
            if let Some(base) = first.as_ref() {
                ensure!(
                    part.version == base.version && part.alignment == base.alignment,
                    "inconsistent GGUF shard version/alignment"
                );
                for (key, value) in &part.meta {
                    if key == "split.no" {
                        continue;
                    }
                    ensure!(
                        base.meta.get(key) == Some(value),
                        "shard metadata disagrees with first shard: {key}"
                    );
                }
            }
            for (name, mut info) in std::mem::take(&mut part.tensors) {
                info.shard_index = index;
                ensure!(
                    !tensors.contains_key(&name),
                    "duplicate tensor across GGUF shards: {name}"
                );
                ensure!(
                    tensors.len() < MAX_ENTRIES,
                    "combined GGUF tensor count exceeds reader limit"
                );
                tensors.insert(name, info);
            }
            if first.is_none() {
                first = Some(part);
            }
            mappings.push(map);
            sources.push(source);
        }
        let first = first.expect("nonempty checked shard count");
        if let Some(declared) = first.meta.get("split.tensors.count") {
            ensure!(
                declared.as_u64() == Some(tensors.len() as u64),
                "combined tensor count does not match split.tensors.count"
            );
        }
        if count > 1 {
            ensure!(!tensors.is_empty(), "GGUF shard set contains no tensors");
        }
        Ok(Self {
            path: paths[0].clone(),
            version: first.version,
            meta: first.meta,
            tensors,
            alignment: first.alignment,
            data_offset: first.data_offset,
            mappings,
            sources,
            paths,
            identity,
            cache: ExpertCache::new(capacity_bytes),
        })
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn expert_cache_stats(&self) -> Result<ExpertCacheStats> {
        self.cache.stats()
    }

    /// Retune expert RAM without replacing the model or invalidating active leases.
    pub fn resize_expert_cache(&self, bytes: usize) -> Result<()> {
        self.cache.resize(bytes)
    }

    /// Configure after header/memory planning, before any expert cache use.
    pub fn set_expert_cache_capacity(&mut self, capacity_bytes: usize) -> Result<()> {
        ensure!(
            self.cache.stats()?.resident_entries == 0,
            "expert cache capacity can only change while empty"
        );
        self.cache = ExpertCache::new(capacity_bytes);
        Ok(())
    }

    /// Warm routed expert triplets only. Sparse per-layer embeddings and dense
    /// weights remain mapped; no whole-model prefault or conversion occurs.
    /// A partial budget is spread across layers, then successive expert IDs.
    pub fn warm_expert_cache(&self) -> Result<ExpertWarmupReport> {
        let started = std::time::Instant::now();
        let mut report = ExpertWarmupReport::default();
        if self.cache.stats()?.capacity_bytes == 0 {
            report.elapsed_seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        let mut layers: BTreeMap<usize, [Option<&str>; 3]> = BTreeMap::new();
        for name in self.tensors.keys() {
            for (kind, suffix) in [
                ".ffn_gate_exps.weight",
                ".ffn_up_exps.weight",
                ".ffn_down_exps.weight",
            ]
            .iter()
            .enumerate()
            {
                if let Some(layer) = name
                    .strip_prefix("blk.")
                    .and_then(|v| v.strip_suffix(suffix))
                {
                    let index: usize = layer
                        .parse()
                        .context("invalid expert tensor layer number")?;
                    layers.entry(index).or_insert([None; 3])[kind] = Some(name);
                }
            }
        }
        let mut geometry = Vec::new();
        for (layer, names) in layers {
            let names = [
                names[0].context("expert warmup missing gate tensor")?,
                names[1].context("expert warmup missing up tensor")?,
                names[2].context("expert warmup missing down tensor")?,
            ];
            let a = self.tensor(names[0])?;
            let b = self.tensor(names[1])?;
            let c = self.tensor(names[2])?;
            ensure!(
                a.dims.len() == 3 && b.dims.len() == 3 && c.dims.len() == 3,
                "layer {layer} expert warmup requires three-dimensional triplets"
            );
            ensure!(
                a.dims == b.dims
                    && a.dims[2] == c.dims[2]
                    && a.dims[0] == c.dims[1]
                    && a.dims[1] == c.dims[0],
                "layer {layer} expert triplet geometry disagrees"
            );
            geometry.push((names, a.dims[2] as usize));
        }
        let total_experts = geometry
            .iter()
            .try_fold(0usize, |n, (_, count)| n.checked_add(*count))
            .context("expert warmup count overflow")?;
        ensure!(
            total_experts <= 1_048_576,
            "expert warmup count exceeds bounded traversal limit"
        );
        let max_experts = geometry.iter().map(|(_, n)| *n).max().unwrap_or(0);
        for expert in 0..max_experts {
            for &(names, count) in &geometry {
                if expert >= count {
                    continue;
                }
                report.experts_considered += 1;
                let matrices = [
                    self.tensor(names[0])?.expert(expert)?,
                    self.tensor(names[1])?.expert(expert)?,
                    self.tensor(names[2])?.expert(expert)?,
                ];
                let ranges: Vec<_> = matrices
                    .iter()
                    .map(|t| (t.cache_key(), t.bytes().len()))
                    .collect();
                let filled = self
                    .cache
                    .warm_batch(&ranges, |key, len| {
                        self.sources
                            .get(key.shard as usize)
                            .context("warmup source shard missing")?
                            .read(key.offset, len)
                    })
                    .with_context(|| format!("warm expert {expert} for {}", names[0]))?;
                match filled {
                    Some((loaded, bytes)) => {
                        report.experts_ready += 1;
                        report.matrices_loaded += loaded;
                        report.payload_bytes_loaded += bytes;
                    }
                    None => report.experts_skipped_budget += 1,
                }
            }
        }
        report.elapsed_seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    /// Opportunistic all-or-nothing host-cache prefetch for a predicted expert.
    /// Never evicts a resident or leased entry, never reads embedding tables,
    /// and does not change which experts the model evaluates.
    /// Ok(None) means insufficient free cache budget; this is not an error.
    pub fn prefetch_expert_triplet(
        &self,
        layer: usize,
        expert: usize,
    ) -> Result<Option<(usize, usize)>> {
        let names = [
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
        ];
        let mut ranges = Vec::with_capacity(3);
        for name in &names {
            let matrix = self.tensor(name)?.expert(expert)?;
            ranges.push((matrix.cache_key(), matrix.bytes().len()));
        }
        self.cache.warm_batch(&ranges, |key, length| {
            self.sources
                .get(key.shard as usize)
                .context("prefetch source shard missing")?
                .read(key.offset, length)
        })
    }

    /// A lease always identifies the same expert bytes, whether owned in the
    /// cache or borrowed directly from its shard. Active leases prevent eviction.
    pub fn expert_cached(&self, name: &str, expert: usize) -> Result<ExpertLease<'_>> {
        let mapped = self.tensor(name)?.expert(expert)?;
        let key = mapped.cache_key();
        let source = self
            .sources
            .get(key.shard as usize)
            .context("expert source shard is missing")?;
        let cached = self.cache.lease_with(key, mapped.bytes().len(), || {
            source.read(key.offset, mapped.bytes().len())
        })?;
        Ok(ExpertLease { mapped, cached })
    }

    pub fn expert_cached_for_prefill(&self, name: &str, expert: usize) -> Result<ExpertLease<'_>> {
        let mapped = self.tensor(name)?.expert(expert)?;
        let cached = self.cache.prefill_lease(mapped.cache_key())?;
        Ok(ExpertLease { mapped, cached })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let end = info
            .file_offset
            .checked_add(info.byte_len)
            .context("tensor range overflow")?;
        let map = self
            .mappings
            .get(info.shard_index)
            .context("invalid tensor shard index")?;
        let bytes = map
            .get(info.file_offset..end)
            .context("mapped tensor range is invalid")?;
        // Public metadata is convenient to inspect, but safe callers can also
        // edit an exclusively owned Gguf. Revalidate geometry before creating
        // a view so such edits cannot bypass the parser's slice invariants.
        ensure!(
            (1..=4).contains(&info.dims.len()),
            "invalid tensor dimension count"
        );
        let mut elements = 1usize;
        for dimension in &info.dims {
            let dimension =
                usize::try_from(*dimension).context("tensor dimension exceeds address space")?;
            ensure!(dimension > 0, "zero tensor dimension");
            elements = elements
                .checked_mul(dimension)
                .context("tensor element count overflow")?;
        }
        let expected = info
            .ty
            .row_bytes(info.dims[0] as usize)?
            .checked_mul(elements / info.dims[0] as usize)
            .context("tensor byte count overflow")?;
        ensure!(
            expected == bytes.len(),
            "tensor byte count disagrees with its dimensions"
        );
        Ok(Tensor {
            dims: &info.dims,
            ty: info.ty,
            bytes,
            key: TensorKey {
                model: self.identity,
                shard: info.shard_index as u32,
                offset: info.file_offset as u64,
                length: info.byte_len as u64,
                quantization: info.ty.id(),
            },
        })
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mappings.iter().map(|m| m.len()).sum()
    }

    pub fn tensor_bytes(&self) -> usize {
        // Parsed::read checked disjoint in-file ranges, so their sum fits usize.
        self.tensors.values().map(|t| t.byte_len).sum()
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ExpertWarmupReport {
    pub experts_considered: usize,
    pub experts_ready: usize,
    pub experts_skipped_budget: usize,
    pub matrices_loaded: usize,
    pub payload_bytes_loaded: usize,
    pub elapsed_seconds: f64,
}

/// Pinned owned expert payload or a zero-allocation mapped bypass. Keep this
/// lease alive through CPU execution or GPU upload; `tensor()` borrows it.
#[derive(Debug)]
pub struct ExpertLease<'a> {
    mapped: Tensor<'a>,
    cached: Option<Arc<OwnedBytes>>,
}
impl ExpertLease<'_> {
    pub fn tensor(&self) -> Tensor<'_> {
        Tensor {
            dims: self.mapped.dims,
            ty: self.mapped.ty,
            bytes: self
                .cached
                .as_ref()
                .map(|v| v.0.as_slice())
                .unwrap_or(self.mapped.bytes),
            key: self.mapped.key,
        }
    }
    pub fn is_owned(&self) -> bool {
        self.cached.is_some()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a> {
    pub(crate) dims: &'a [u64],
    pub(crate) ty: QuantType,
    bytes: &'a [u8],
    key: TensorKey,
}

impl<'a> Tensor<'a> {
    pub fn dimensions(&self) -> &'a [u64] {
        self.dims
    }

    pub fn cache_key(&self) -> TensorKey {
        self.key
    }

    pub fn quant_type(&self) -> QuantType {
        self.ty
    }

    pub fn width(&self) -> usize {
        self.dims[0] as usize
    }

    pub fn rows(&self) -> usize {
        self.dims[1..].iter().map(|n| *n as usize).product()
    }

    pub fn elements(&self) -> usize {
        self.width() * self.rows()
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    fn row_bytes(&self, row: usize) -> Result<&'a [u8]> {
        ensure!(
            row < self.rows(),
            "tensor row {row} is outside {} rows",
            self.rows()
        );
        let stride = self.ty.row_bytes(self.width())?;
        let start = row
            .checked_mul(stride)
            .context("tensor row offset overflow")?;
        self.bytes
            .get(start..start + stride)
            .context("tensor row is truncated")
    }

    pub fn read_row(&self, row: usize, output: &mut [f32]) -> Result<()> {
        ensure!(
            output.len() == self.width(),
            "tensor row output width mismatch"
        );
        quant::decode_row(self.ty, self.row_bytes(row)?, output)
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        ensure!(row < self.rows(), "tensor row {row} is out of range");
        let mut output = vec![0.0; self.width()];
        self.read_row(row, &mut output)?;
        Ok(output)
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = vec![0.0; self.rows()];
        self.matvec_into(input, &mut output)?;
        Ok(output)
    }

    /// Independent rows can use the process Rayon pool for sufficiently large
    /// work; tiny tensors stay serial. The weight rows are never materialized.
    pub fn matvec_into(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        ensure!(input.len() == self.width(), "matvec input width mismatch");
        ensure!(output.len() == self.rows(), "matvec output height mismatch");
        ensure!(
            input.iter().all(|x| x.is_finite()),
            "non-finite matvec input"
        );
        let compute = |(row, out): (usize, &mut f32)| -> Result<()> {
            *out = quant::dot_finite_input(self.ty, self.row_bytes(row)?, input)?;
            Ok(())
        };
        if output.len() >= 64 && input.len() >= 256 {
            output.par_iter_mut().enumerate().try_for_each(compute)
        } else {
            output.iter_mut().enumerate().try_for_each(compute)
        }
    }

    /// Compatibility helper: direct block dot products need no dequant buffer.
    pub fn matvec_with_scratch(
        &self,
        input: &[f32],
        output: &mut [f32],
        _row: &mut Vec<f32>,
    ) -> Result<()> {
        self.matvec_into(input, output)
    }

    /// A routed expert is a zero-copy 2D view of a 3D GGML tensor.
    pub fn expert(&self, expert: usize) -> Result<Tensor<'a>> {
        ensure!(
            self.dims.len() == 3,
            "expert view requires a three-dimensional tensor"
        );
        ensure!(
            expert < self.dims[2] as usize,
            "expert index {expert} is out of range"
        );
        let size = self
            .ty
            .row_bytes(self.width())?
            .checked_mul(self.dims[1] as usize)
            .context("expert tensor size overflow")?;
        let start = expert
            .checked_mul(size)
            .context("expert tensor offset overflow")?;
        let bytes = self
            .bytes
            .get(start..start + size)
            .context("expert tensor is truncated")?;
        let mut key = self.key;
        key.offset = key
            .offset
            .checked_add(start as u64)
            .context("expert identity offset overflow")?;
        key.length = size as u64;
        Ok(Tensor {
            dims: &self.dims[..2],
            ty: self.ty,
            bytes,
            key,
        })
    }
}

fn split_integer(meta: &BTreeMap<String, Value>, name: &str, default: usize) -> Result<usize> {
    match meta.get(name) {
        None => Ok(default),
        Some(value) => usize::try_from(value.as_u64().with_context(|| format!("invalid {name}"))?)
            .with_context(|| format!("{name} exceeds addressable range")),
    }
}

fn map_file(path: &Path) -> Result<(Arc<Mmap>, Parsed, expert_io::Source)> {
    let file = File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
    let len = file.metadata().context("read model metadata")?.len();
    ensure!(
        (24..=isize::MAX as u64).contains(&len),
        "GGUF file is truncated or exceeds address space"
    );
    // SAFETY: immutable mapping remains owned by Gguf for every borrowed view.
    // The model file must not be edited or truncated while mapped.
    let mapping = Arc::new(unsafe { MmapOptions::new().map(&file) }.context("map GGUF shard")?);
    let parsed =
        Parsed::read(&mapping).with_context(|| format!("parse GGUF {}", path.display()))?;
    let source = expert_io::Source::new(path, file, len);
    Ok((mapping, parsed, source))
}

fn shard_paths(path: &Path, count: usize, number: usize) -> Result<Vec<PathBuf>> {
    ensure!(
        (1..=MAX_SHARDS).contains(&count) && number < count,
        "invalid GGUF shard count/ordinal"
    );
    if count == 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("sharded GGUF filename must be UTF-8")?;
    let suffix = format!("-{:05}-of-{:05}.gguf", number + 1, count);
    let prefix = name
        .strip_suffix(&suffix)
        .context("split GGUF needs its standard five-digit shard filename matching metadata")?;
    ensure!(!prefix.is_empty(), "empty GGUF shard filename prefix");
    Ok((1..=count)
        .map(|n| path.with_file_name(format!("{prefix}-{n:05}-of-{count:05}.gguf")))
        .collect())
}

struct Parsed {
    version: u32,
    meta: BTreeMap<String, Value>,
    tensors: BTreeMap<String, TensorInfo>,
    alignment: usize,
    data_offset: usize,
}

impl Parsed {
    fn read(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader {
            bytes,
            cursor: 0,
            allocated: 0,
        };
        ensure!(reader.take(4)? == b"GGUF", "not a little-endian GGUF file");
        let version = reader.u32()?;
        ensure!(
            (2..=3).contains(&version),
            "unsupported GGUF version {version}; expected little-endian v2 or v3"
        );
        let n_tensors =
            usize::try_from(reader.u64()?).context("GGUF tensor count exceeds address space")?;
        let n_meta =
            usize::try_from(reader.u64()?).context("GGUF metadata count exceeds address space")?;
        ensure!(
            n_tensors <= MAX_ENTRIES && n_meta <= MAX_ENTRIES,
            "GGUF header entry count exceeds reader limits"
        );
        ensure!(
            n_meta <= reader.remaining() / 13,
            "GGUF metadata count exceeds remaining header bytes"
        );
        let mut meta = BTreeMap::new();
        for _ in 0..n_meta {
            let key = reader.string(65_535)?;
            ensure!(
                !key.is_empty() && key.is_ascii() && !key.as_bytes().contains(&0),
                "invalid GGUF metadata key"
            );
            ensure!(
                !meta.contains_key(&key),
                "duplicate GGUF metadata key {key}"
            );
            let ty = reader.u32()?;
            let value = reader
                .value(ty, 0)
                .with_context(|| format!("read metadata {key}"))?;
            meta.insert(key, value);
        }
        let count = split_integer(&meta, "split.count", 1)?;
        let number = split_integer(&meta, "split.no", 0)?;
        ensure!(
            (1..=MAX_SHARDS).contains(&count) && number < count,
            "invalid GGUF split count/ordinal"
        );
        if count > 1 {
            ensure!(
                meta.contains_key("split.no") && meta.contains_key("split.tensors.count"),
                "split GGUF needs ordinal and aggregate tensor count"
            );
        }
        if let Some(total) = meta.get("split.tensors.count") {
            let total = total.as_u64().context("invalid split.tensors.count")?;
            ensure!(
                total >= n_tensors as u64 && total <= MAX_ENTRIES as u64,
                "invalid aggregate split tensor count"
            );
            if count == 1 {
                ensure!(
                    total == n_tensors as u64,
                    "single-file tensor count disagrees"
                );
            }
        }
        let alignment = match meta.get("general.alignment") {
            None => 32,
            Some(value) => usize::try_from(
                value
                    .as_u64()
                    .context("general.alignment must be an unsigned integer")?,
            )
            .context("GGUF alignment exceeds address space")?,
        };
        ensure!(
            alignment > 0 && alignment % 8 == 0 && alignment <= MAX_ALIGNMENT,
            "GGUF alignment must be a multiple of 8 between 8 and {MAX_ALIGNMENT}"
        );
        // Each descriptor has at least a string length, dimension count, one
        // dimension, type and offset. This bound precedes any descriptor allocation.
        ensure!(
            n_tensors <= reader.remaining() / 32,
            "GGUF tensor count exceeds remaining descriptor bytes"
        );
        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = reader.string(64)?;
            ensure!(
                !name.is_empty() && !name.as_bytes().contains(&0),
                "invalid GGUF tensor name"
            );
            ensure!(
                !tensors.contains_key(&name),
                "duplicate GGUF tensor name {name}"
            );
            let dimensions = reader.u32()? as usize;
            ensure!(
                (1..=4).contains(&dimensions),
                "tensor {name} has unsupported dimension count {dimensions}"
            );
            let mut dims = Vec::with_capacity(dimensions);
            let mut elements = 1usize;
            for _ in 0..dimensions {
                let length = reader.u64()?;
                ensure!(length > 0, "tensor {name} has a zero dimension");
                let local =
                    usize::try_from(length).context("tensor dimension exceeds address space")?;
                elements = elements
                    .checked_mul(local)
                    .context("tensor element count overflow")?;
                dims.push(length);
            }
            let ty =
                QuantType::try_from(reader.u32()?).with_context(|| format!("tensor {name}"))?;
            let offset = reader.u64()?;
            ensure!(
                offset % alignment as u64 == 0,
                "tensor {name} offset is not aligned to {alignment}"
            );
            let row_size = ty
                .row_bytes(dims[0] as usize)
                .with_context(|| format!("tensor {name}"))?;
            let byte_len = row_size
                .checked_mul(elements / dims[0] as usize)
                .context("tensor byte count overflow")?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ty,
                    offset,
                    byte_len,
                    file_offset: 0,
                    shard_index: 0,
                },
            );
        }
        let padding = (alignment - reader.cursor % alignment) % alignment;
        let data_offset = reader
            .cursor
            .checked_add(padding)
            .context("GGUF data offset overflow")?;
        ensure!(
            data_offset <= bytes.len() || tensors.is_empty(),
            "GGUF tensor-data padding is truncated"
        );
        let mut ranges = Vec::with_capacity(tensors.len());
        for tensor in tensors.values_mut() {
            let offset =
                usize::try_from(tensor.offset).context("tensor offset exceeds address space")?;
            tensor.file_offset = data_offset
                .checked_add(offset)
                .context("tensor file offset overflow")?;
            let end = tensor
                .file_offset
                .checked_add(tensor.byte_len)
                .context("tensor range overflow")?;
            ensure!(
                end <= bytes.len(),
                "tensor {} is truncated: requires byte {end}, file has {}",
                tensor.name,
                bytes.len()
            );
            ranges.push((tensor.file_offset, end, tensor.name.as_str()));
        }
        ranges.sort_unstable_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "GGUF tensor ranges overlap: {} and {}",
                pair[0].2,
                pair[1].2
            );
        }
        Ok(Self {
            version,
            meta,
            tensors,
            alignment,
            data_offset,
        })
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    allocated: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .cursor
            .checked_add(count)
            .context("GGUF cursor overflow")?;
        ensure!(
            end <= MAX_HEADER_BYTES,
            "GGUF header exceeds {MAX_HEADER_BYTES} byte reader limit"
        );
        let data = self
            .bytes
            .get(self.cursor..end)
            .context("truncated GGUF header")?;
        self.cursor = end;
        Ok(data)
    }

    fn allocation(&mut self, count: usize) -> Result<()> {
        self.allocated = self
            .allocated
            .checked_add(count)
            .context("GGUF metadata allocation overflow")?;
        ensure!(
            self.allocated <= MAX_METADATA_ALLOCATIONS,
            "GGUF metadata exceeds bounded reader memory"
        );
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn string(&mut self, limit: usize) -> Result<String> {
        let count =
            usize::try_from(self.u64()?).context("GGUF string length exceeds address space")?;
        ensure!(
            count <= limit && count <= MAX_STRING_BYTES,
            "GGUF string exceeds bounded reader length"
        );
        self.allocation(count)?;
        let text = std::str::from_utf8(self.take(count)?).context("GGUF string is not UTF-8")?;
        Ok(text.to_owned())
    }

    fn value(&mut self, ty: u32, depth: usize) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(match self.u8()? {
                0 => false,
                1 => true,
                other => bail!("invalid GGUF boolean byte {other}"),
            }),
            8 => Value::String(self.string(MAX_STRING_BYTES)?),
            9 => {
                ensure!(
                    depth < MAX_ARRAY_DEPTH,
                    "GGUF metadata arrays exceed nesting limit"
                );
                let element_type = self.u32()?;
                let count = usize::try_from(self.u64()?)
                    .context("GGUF array count exceeds address space")?;
                ensure!(
                    count <= MAX_ARRAY_ITEMS,
                    "GGUF array exceeds bounded reader length"
                );
                let minimum_size = match element_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4 | 5 | 6 => 4,
                    8 | 10 | 11 | 12 => 8,
                    9 => 12,
                    other => bail!("unsupported GGUF metadata array type {other}"),
                };
                ensure!(
                    count <= self.remaining() / minimum_size,
                    "GGUF array count exceeds remaining data"
                );
                self.allocation(
                    count
                        .checked_mul(std::mem::size_of::<Value>())
                        .context("GGUF array allocation overflow")?,
                )?;
                let mut values = Vec::new();
                values
                    .try_reserve_exact(count)
                    .context("allocate bounded GGUF metadata array")?;
                for _ in 0..count {
                    values.push(self.value(element_type, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => bail!("unsupported GGUF metadata value type {other}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn text(out: &mut Vec<u8>, value: &str) {
        out.extend((value.len() as u64).to_le_bytes());
        out.extend(value.as_bytes());
    }

    fn fixture(
        metadata: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], u32, u64)],
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend((tensors.len() as u64).to_le_bytes());
        out.extend((metadata.len() as u64).to_le_bytes());
        for (name, ty, bytes) in metadata {
            text(&mut out, name);
            out.extend(ty.to_le_bytes());
            out.extend(bytes);
        }
        for (name, dims, ty, offset) in tensors {
            text(&mut out, name);
            out.extend((dims.len() as u32).to_le_bytes());
            for length in *dims {
                out.extend(length.to_le_bytes());
            }
            out.extend(ty.to_le_bytes());
            out.extend(offset.to_le_bytes());
        }
        out.resize(out.len().next_multiple_of(32), 0);
        out.extend(data);
        out
    }

    struct Temp(PathBuf);
    impl Temp {
        fn write(bytes: &[u8]) -> Self {
            let id = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("forge-gguf-{}-{id}.gguf", std::process::id()));
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::io::Write::write_all(&mut file, bytes).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn mapped_expert_views_keep_exact_rows_and_do_not_expand_weights() {
        let weights: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let bytes: Vec<u8> = weights.iter().flat_map(|x| x.to_le_bytes()).collect();
        let file = Temp::write(&fixture(&[], &[("experts", &[2, 3, 2], 0, 0)], &bytes));
        let model = Arc::new(Gguf::open(&file.0).unwrap());
        assert_eq!(model.tensor_bytes(), bytes.len());
        let tensor = model.tensor("experts").unwrap();
        let expert = tensor.expert(1).unwrap();
        assert_eq!(expert.dims, &[2, 3]);
        assert_eq!(expert.row(1).unwrap(), vec![8.0, 9.0]);
        assert_eq!(expert.matvec(&[1.0, -1.0]).unwrap(), vec![-1.0; 3]);
        assert_eq!(expert.bytes().as_ptr(), tensor.bytes()[24..].as_ptr());
        assert!(tensor.expert(2).is_err());
        assert!(expert.row(3).is_err());
        assert!(expert.matvec(&[1.0]).is_err());
    }

    #[test]
    fn truncated_files_and_unbounded_counts_are_rejected() {
        let data = fixture(&[], &[("w", &[2], 0, 0)], &[0; 8]);
        for length in [0, 3, 7, 15, 23, data.len() - 1] {
            assert!(
                Parsed::read(&data[..length]).is_err(),
                "accepted truncation at {length}"
            );
        }
        let mut huge_count = data.clone();
        huge_count[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Parsed::read(&huge_count).is_err());
        let oversized = fixture(&[], &[("w", &[u64::MAX, 2], 0, 0)], &[]);
        assert!(Parsed::read(&oversized).is_err());
        let overflow_offset = fixture(&[], &[("w", &[2], 0, u64::MAX - 31)], &[0; 8]);
        assert!(Parsed::read(&overflow_offset).is_err());
    }

    #[test]
    fn duplicates_overlaps_and_unaligned_offsets_are_rejected() {
        let duplicates = fixture(&[("x", 4, vec![0; 4]), ("x", 4, vec![0; 4])], &[], &[]);
        assert!(Parsed::read(&duplicates).is_err());
        let duplicate_tensors = fixture(&[], &[("w", &[2], 0, 0), ("w", &[2], 0, 32)], &[0; 40]);
        assert!(Parsed::read(&duplicate_tensors).is_err());
        let overlap = fixture(&[], &[("a", &[16], 0, 0), ("b", &[2], 0, 32)], &[0; 64]);
        assert!(Parsed::read(&overlap).is_err());
        let unaligned = fixture(&[], &[("w", &[2], 0, 1)], &[0; 40]);
        assert!(Parsed::read(&unaligned).is_err());
    }

    #[test]
    fn unsupported_formats_shards_and_invalid_metadata_fail_explicitly() {
        let unsupported = fixture(&[], &[("w", &[32], 43, 0)], &[0; 128]);
        let error = Parsed::read(&unsupported).err().unwrap();
        assert!(format!("{error:#}").contains("unsupported"));
        for count in [0u32, 2] {
            let split = fixture(
                &[("split.count", 4, count.to_le_bytes().to_vec())],
                &[],
                &[],
            );
            assert!(Parsed::read(&split).is_err());
        }
        let bad_boolean = fixture(&[("bool", 7, vec![2])], &[], &[]);
        assert!(Parsed::read(&bad_boolean).is_err());
        let bad_alignment = fixture(
            &[("general.alignment", 4, 0u32.to_le_bytes().to_vec())],
            &[],
            &[],
        );
        assert!(Parsed::read(&bad_alignment).is_err());
        let incomplete_block = fixture(&[], &[("w", &[31], 2, 0)], &[0; 18]);
        assert!(Parsed::read(&incomplete_block).is_err());
    }

    #[test]
    fn arrays_are_bounded_and_nested_values_remain_typed() {
        let mut array = 9u32.to_le_bytes().to_vec();
        array.extend(1u64.to_le_bytes());
        array.extend(4u32.to_le_bytes());
        array.extend(2u64.to_le_bytes());
        array.extend(7u32.to_le_bytes());
        array.extend(8u32.to_le_bytes());
        let data = fixture(&[("nested", 9, array)], &[], &[]);
        let parsed = Parsed::read(&data).unwrap();
        let outer = parsed.meta["nested"].as_array().unwrap();
        assert_eq!(
            outer[0].as_array().unwrap(),
            &[Value::U32(7), Value::U32(8)]
        );
        let mut huge = 4u32.to_le_bytes().to_vec();
        huge.extend(u64::MAX.to_le_bytes());
        assert!(Parsed::read(&fixture(&[("large", 9, huge)], &[], &[])).is_err());
        assert_eq!(Value::I32(-1).as_u64(), None);
        assert_eq!(Value::F64(f64::MAX).as_f32(), None);
        assert_eq!(Value::F64(f64::MIN_POSITIVE).as_f32(), None);
    }

    #[test]
    fn mapped_quantized_matrix_matches_hand_computed_rows() {
        let mut bytes = Vec::new();
        for scale in [1.0, 0.5] {
            bytes.extend(half::f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend((0..32).map(|i| (i as i8 - 16) as u8));
        }
        let file = Temp::write(&fixture(&[], &[("w", &[32, 2], 8, 0)], &bytes));
        let mapped = Gguf::open(&file.0).unwrap();
        let matrix = mapped.tensor("w").unwrap();
        assert_eq!(matrix.matvec(&[1.0; 32]).unwrap(), vec![-16.0, -8.0]);
        let mut output = [0.0; 2];
        matrix.matvec_into(&[0.0; 32], &mut output).unwrap();
        assert_eq!(output, [0.0, 0.0]);
    }

    #[test]
    fn large_rows_use_parallel_dispatch_with_same_results() {
        let mut weights = Vec::new();
        for row in 0..64 {
            weights.extend((0..256).flat_map(|col| ((row + col % 3) as f32 * 0.25).to_le_bytes()));
        }
        let file = Temp::write(&fixture(&[], &[("w", &[256, 64], 0, 0)], &weights));
        let mapped = Gguf::open(&file.0).unwrap();
        let matrix = mapped.tensor("w").unwrap();
        let input: Vec<f32> = (0..256)
            .map(|col| if col % 2 == 0 { 1.0 } else { -0.5 })
            .collect();
        let actual = matrix.matvec(&input).unwrap();
        for (row, output) in actual.iter().enumerate() {
            let expected =
                quant::dot_scalar(QuantType::F32, matrix.row_bytes(row).unwrap(), &input).unwrap();
            assert_eq!(*output, expected);
        }
    }

    #[test]
    fn caller_metadata_edits_cannot_construct_an_invalid_view() {
        let file = Temp::write(&fixture(&[], &[("w", &[2], 0, 0)], &[0; 8]));
        let mut mapped = Gguf::open(&file.0).unwrap();
        mapped.tensors.get_mut("w").unwrap().dims.clear();
        assert!(mapped.tensor("w").is_err());
    }
    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let id = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("forge-shards-{}-{id}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn shard(&self, ordinal: usize) -> PathBuf {
            self.0.join(format!("model-{ordinal:05}-of-00002.gguf"))
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn shard_metadata(
        ordinal: u32,
        total: u64,
        architecture: Option<&str>,
    ) -> Vec<(&'static str, u32, Vec<u8>)> {
        let mut values = vec![
            ("split.count", 4, 2u32.to_le_bytes().to_vec()),
            ("split.no", 4, ordinal.to_le_bytes().to_vec()),
            ("split.tensors.count", 10, total.to_le_bytes().to_vec()),
        ];
        if let Some(architecture) = architecture {
            let mut value = Vec::new();
            text(&mut value, architecture);
            values.push(("general.architecture", 8, value));
        }
        values
    }

    #[test]
    fn complete_shards_allow_metadata_only_first_and_open_from_any_part() {
        let dir = TempDir::new();
        let first = fixture(&shard_metadata(0, 1, Some("qwen3moe")), &[], &[]);
        let values = [1f32, -2., 3., 4.];
        let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let second = fixture(&shard_metadata(1, 1, None), &[("w", &[2, 2], 0, 0)], &bytes);
        std::fs::write(dir.shard(1), &first).unwrap();
        std::fs::write(dir.shard(2), &second).unwrap();
        let model = Gguf::open(dir.shard(2)).unwrap();
        assert_eq!(model.path, dir.shard(1));
        assert_eq!(model.shard_paths().len(), 2);
        assert_eq!(model.mapped_bytes(), first.len() + second.len());
        assert_eq!(model.tensor_bytes(), bytes.len());
        assert_eq!(
            model.tensor("w").unwrap().matvec(&[2., 1.]).unwrap(),
            [0., 10.]
        );
        assert_eq!(model.tensor("w").unwrap().cache_key().shard, 1);
    }

    #[test]
    fn shard_set_rejects_missing_duplicate_mismatched_and_inconsistent_parts() {
        let dir = TempDir::new();
        let a = fixture(
            &shard_metadata(0, 2, Some("qwen3moe")),
            &[("a", &[2], 0, 0)],
            &[0; 8],
        );
        std::fs::write(dir.shard(1), &a).unwrap();
        assert!(Gguf::open(dir.shard(1)).is_err());
        for (metadata, name) in [
            (shard_metadata(1, 2, None), "a"), // duplicate across files
            (shard_metadata(0, 2, None), "b"), // wrong ordinal
            (shard_metadata(1, 3, None), "b"), // aggregate mismatch
            (shard_metadata(1, 2, Some("other")), "b"), // conflicting model
        ] {
            std::fs::write(
                dir.shard(2),
                fixture(&metadata, &[(name, &[2], 0, 0)], &[0; 8]),
            )
            .unwrap();
            assert!(
                Gguf::open(dir.shard(1)).is_err(),
                "accepted {name} {metadata:?}"
            );
        }
        std::fs::write(
            dir.shard(2),
            fixture(&shard_metadata(1, 2, None), &[("b", &[2], 0, 0)], &[0; 8]),
        )
        .unwrap();
        assert_eq!(Gguf::open(dir.shard(1)).unwrap().tensors.len(), 2);
        let nonstandard = dir.0.join("renamed.gguf");
        std::fs::write(&nonstandard, a).unwrap();
        assert!(Gguf::open(&nonstandard).is_err());
    }

    #[test]
    fn owned_expert_leases_preserve_bytes_identity_and_hard_capacity() {
        let values: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let data: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let file = Temp::write(&fixture(&[], &[("experts", &[2, 3, 2], 0, 0)], &data));
        let model = Gguf::open_with_cache(&file.0, 24).unwrap();
        let first = model.expert_cached("experts", 0).unwrap();
        assert!(first.is_owned());
        let first_key = first.tensor().cache_key();
        assert_eq!(
            first_key,
            model
                .tensor("experts")
                .unwrap()
                .expert(0)
                .unwrap()
                .cache_key()
        );
        let second_bypass = model.expert_cached("experts", 1).unwrap();
        assert!(!second_bypass.is_owned());
        assert_eq!(second_bypass.tensor().row(1).unwrap(), [8., 9.]);
        assert_eq!(model.expert_cache_stats().unwrap().pinned_bytes, 24);
        drop(first);
        let second = model.expert_cached("experts", 1).unwrap();
        assert!(second.is_owned());
        assert_eq!(
            second.tensor().cache_key(),
            second_bypass.tensor().cache_key()
        );
        assert_ne!(first_key, second.tensor().cache_key());
        assert_eq!(model.expert_cache_stats().unwrap().resident_bytes, 24);
        assert_eq!(model.expert_cache_stats().unwrap().evictions, 1);
        let stats = model.expert_cache_stats().unwrap();
        assert_eq!(stats.direct_expert_loads + stats.buffered_expert_loads, 2);
        assert_eq!(stats.payload_bytes_read, 48);
        assert_eq!(stats.read_errors, 0);
        let before = stats.resident_bytes;
        let prompt_miss = model.expert_cached_for_prefill("experts", 0).unwrap();
        assert!(!prompt_miss.is_owned());
        assert!(
            model
                .expert_cached_for_prefill("experts", 1)
                .unwrap()
                .is_owned()
        );
        let stats = model.expert_cache_stats().unwrap();
        assert_eq!(
            (
                stats.prefill_hits,
                stats.prefill_misses,
                stats.resident_bytes
            ),
            (1, 1, before)
        );
        assert_eq!(stats.evictions, 1);
        let reopened = Gguf::open(&file.0).unwrap();
        assert_ne!(
            model.tensor("experts").unwrap().cache_key(),
            reopened.tensor("experts").unwrap().cache_key()
        );
        assert!(model.expert_cached("experts", 2).is_err());
    }

    #[test]
    fn warmup_admits_complete_triplets_without_eviction_or_ple_reads() {
        let weights: Vec<u8> = (0..48).flat_map(|i| (i as f32).to_le_bytes()).collect();
        // Expert triplet payload=16+16+16 bytes per expert. Last tensor is PLE.
        let file = Temp::write(&fixture(
            &[],
            &[
                ("blk.0.ffn_gate_exps.weight", &[2, 2, 2], 0, 0),
                ("blk.0.ffn_up_exps.weight", &[2, 2, 2], 0, 32),
                ("blk.0.ffn_down_exps.weight", &[2, 2, 2], 0, 64),
                ("per_layer_token_embd.weight", &[2, 12], 0, 96),
            ],
            &weights,
        ));
        let mut model = Gguf::open(&file.0).unwrap();
        model.set_expert_cache_capacity(64).unwrap();
        let warmed = model.warm_expert_cache().unwrap();
        assert_eq!(
            (
                warmed.experts_considered,
                warmed.experts_ready,
                warmed.experts_skipped_budget
            ),
            (2, 1, 1)
        );
        assert_eq!(
            (warmed.matrices_loaded, warmed.payload_bytes_loaded),
            (3, 48)
        );
        let stats = model.expert_cache_stats().unwrap();
        assert_eq!(
            (
                stats.resident_bytes,
                stats.resident_entries,
                stats.evictions
            ),
            (48, 3, 0)
        );
        assert_eq!(stats.direct_expert_loads + stats.buffered_expert_loads, 3);
        assert!(
            model
                .expert_cached_for_prefill("blk.0.ffn_gate_exps.weight", 0)
                .unwrap()
                .is_owned()
        );
        assert!(
            !model
                .expert_cached_for_prefill("blk.0.ffn_gate_exps.weight", 1)
                .unwrap()
                .is_owned()
        );
        let again = model.warm_expert_cache().unwrap();
        assert_eq!((again.matrices_loaded, again.payload_bytes_loaded), (0, 0));
        assert_eq!(model.expert_cache_stats().unwrap().resident_bytes, 48);
        assert!(model.set_expert_cache_capacity(128).is_err());
        let disabled = Gguf::open(&file.0).unwrap();
        assert_eq!(disabled.warm_expert_cache().unwrap().experts_considered, 0);
    }
}
