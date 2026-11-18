use clap::Parser;

#[derive(Parser)]
#[command(
    version,
    about = "Independent Forge CPU inference reference; GPU/Qwen3.8 are not implemented"
)]
struct Cli {
    #[command(flatten)]
    engine: forge_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    forge_engine::cli::run(Cli::parse().engine)
}
