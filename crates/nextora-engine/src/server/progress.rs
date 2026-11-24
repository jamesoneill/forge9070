//! A short-lived lock protects the UI snapshot, independent of model and GPU locks.
use super::*;
use crate::runtime::LoadStep;
use std::borrow::Cow;

struct Counts {
    done: usize,
    total: usize,
    unit: &'static str,
    bytes: u64,
    experts_ready: Option<usize>,
}

pub(super) struct LoadProgress {
    stage: &'static str,
    message: Cow<'static, str>,
    started: Option<Instant>,
    elapsed: Option<Duration>,
    counts: Option<Counts>,
}

impl Default for LoadProgress {
    fn default() -> Self {
        Self {
            stage: "idle",
            message: "Waiting for a model".into(),
            started: None,
            elapsed: None,
            counts: None,
        }
    }
}

impl LoadProgress {
    pub(super) fn begin(&mut self) {
        self.started = Some(Instant::now());
        self.elapsed = None;
        self.update(LoadStep::Stage(
            "validating",
            "Checking settings and model files",
        ));
    }

    pub(super) fn update(&mut self, step: LoadStep) {
        self.counts = None;
        match step {
            LoadStep::Stage(stage, message) => {
                self.stage = stage;
                self.message = message.into();
            }
            LoadStep::Upload {
                message,
                unit,
                completed,
                total,
                bytes,
            } => {
                self.stage = "uploading";
                self.message = message.into();
                self.counts = Some(Counts {
                    done: completed,
                    total,
                    unit,
                    bytes,
                    experts_ready: None,
                });
            }
            LoadStep::Warmup {
                considered,
                total,
                ready,
                bytes,
            } => {
                self.stage = "warming";
                self.message = "Warming the expert cache".into();
                self.counts = Some(Counts {
                    done: considered,
                    total,
                    unit: "experts",
                    bytes: bytes as u64,
                    experts_ready: Some(ready),
                });
            }
        }
    }

    pub(super) fn finish(&mut self, stage: &'static str, message: impl Into<Cow<'static, str>>) {
        self.elapsed = self.started.map(|start| start.elapsed());
        self.stage = stage;
        self.message = message.into();
        self.counts = None;
    }

    pub(super) fn snapshot(&self) -> Value {
        let elapsed = self
            .elapsed
            .or_else(|| self.started.map(|start| start.elapsed()));
        let counts = self.counts.as_ref();
        json!({
            "stage": self.stage, "message": self.message,
            "elapsed_seconds": elapsed.unwrap_or_default().as_secs_f64(),
            "completed": counts.map(|c| c.done),
            "total": counts.map(|c| c.total),
            "unit": counts.map(|c| c.unit),
            "experts_ready": counts.and_then(|c| c.experts_ready),
            "bytes_loaded": counts.map(|c| c.bytes),
        })
    }
}
