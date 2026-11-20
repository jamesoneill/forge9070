//! Historical forge9070 command name forwards to the SAME independent Rust runtime.
use clap::Parser;

#[derive(Parser)]
#[command(version, about = "ForgeAI native Rust inference (legacy forge9070 command alias)")]
struct Cli {
    #[command(flatten)]
    engine: forge_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    forge_engine::cli::run(Cli::parse().engine)
}
