//! Historical nextora-cli command name forwards to the SAME independent Rust runtime.
use clap::Parser;

#[derive(Parser)]
#[command(version, about = "Nextora native Rust inference (legacy nextora-cli command alias)")]
struct Cli {
    #[command(flatten)]
    engine: nextora_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    nextora_engine::cli::run(Cli::parse().engine)
}
