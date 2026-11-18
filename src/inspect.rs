use anyhow::{Context, Result};
use rusting_strata::compatibility::{self, Policy, Report};
use serde::Serialize;
use std::{ops::Deref, path::Path};

/// The same conservative MoE policy is used by the HF picker and every load.
#[derive(Debug, Serialize)]
pub struct ModelReport {
    pub path: String,
    #[serde(flatten)]
    pub compatibility: Report,
}
impl Deref for ModelReport {
    type Target = Report;
    fn deref(&self) -> &Report { &self.compatibility }
}
impl ModelReport {
    pub fn ensure_supported(&self) -> Result<()> { self.compatibility.ensure_supported() }
    pub fn print_human(&self) {
        println!("Model: {}\nArchitecture: {}\nLayers: {}\nExperts per layer / active: {} / {}\nShards: {}\nTensors: {}\nWeight payload: {:.3} GiB",
            self.path, self.architecture, self.layers, self.experts_per_layer, self.active_experts_per_token,
            self.shards, self.tensors, self.tensor_weight_bytes as f64 / 1_073_741_824.0);
        println!("Routed experts: {:.3} GiB | lazy embeddings: {:.3} GiB | other weights: {:.3} GiB",
            self.weight_memory.routed_expert_bytes as f64 / 1_073_741_824.0,
            self.weight_memory.lazy_embedding_bytes as f64 / 1_073_741_824.0,
            self.weight_memory.other_weight_bytes as f64 / 1_073_741_824.0);
        for (kind, count) in &self.quantization_histogram { println!("  {kind}: {count} tensors"); }
        println!("MoE metadata preflight: {}", if self.supported_metadata { "accepted" } else { "rejected" });
        for issue in &self.issues { println!("  {issue}"); }
        println!("{}", self.caveat);
    }
}
pub fn inspect(path: &Path) -> Result<ModelReport> {
    let compatibility = compatibility::inspect_path(path, Policy::MoeOnly)
        .with_context(|| format!("inspect {}", path.display()))?;
    Ok(ModelReport { path: path.display().to_string(), compatibility })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3moe.gguf")
    }
    #[test]
    fn reports_real_moe_payload_and_routing_geometry() {
        let report = inspect(&fixture()).unwrap();
        assert_eq!(report.architecture, "qwen3moe");
        assert_eq!((report.experts_per_layer, report.active_experts_per_token), (4,2));
        assert_eq!(report.tensors, 15);
        assert_eq!(report.tensor_weight_bytes, 182656);
        assert_eq!(report.weight_memory.routed_expert_bytes + report.weight_memory.lazy_embedding_bytes
            + report.weight_memory.other_weight_bytes, report.tensor_weight_bytes);
        assert!(report.weight_memory.routed_expert_bytes > 0);
        assert_eq!(report.weight_memory.lazy_embedding_bytes, 0);
        report.ensure_supported().unwrap();
    }
    #[test]
    fn dense_models_are_outside_the_requested_moe_policy() {
        let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3.gguf");
        let report=inspect(&path).unwrap();
        assert!(report.ensure_supported().is_err());
        compatibility::inspect_path(&path,Policy::Supported).unwrap().ensure_supported().unwrap();
    }
    #[test]
    fn unsupported_quantization_is_rejected_before_engine_loading() {
        let mut bytes = std::fs::read(fixture()).unwrap();
        let name = b"token_embd.weight";
        let at = bytes.windows(name.len()).position(|s| s == name).unwrap() + name.len();
        let dims = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind = at + 4 + dims * 8;
        bytes[kind..kind + 4].copy_from_slice(&999_u32.to_le_bytes());
        let path = std::env::temp_dir().join(format!("forge9070-{}-unsupported.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let report = inspect(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(report.quantization_histogram["UNSUPPORTED_ID_999"], 1);
        assert!(report.ensure_supported().is_err());
    }
    #[test]
    fn avalon_custom_filename_is_conservatively_rejected() {
        let path = std::env::temp_dir().join(format!("forge9070-{}-IQ3_KS.gguf", std::process::id()));
        std::fs::copy(fixture(), &path).unwrap();
        let report = inspect(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(report.ensure_supported().is_err());
        assert!(report.issues.iter().any(|s| s.to_lowercase().contains("iq3")));
    }
}
