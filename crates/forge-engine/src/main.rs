use clap::Parser;

#[derive(Parser)]
#[command(
    version,
    about = "Independent Forge inference with capability-selected Vulkan, CPU and diagnostic hybrid execution"
)]
struct Cli {
    #[command(flatten)]
    engine: forge_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    forge_engine::cli::run(Cli::parse().engine)
}
