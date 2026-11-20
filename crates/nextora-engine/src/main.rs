use clap::Parser;

#[derive(Parser)]
#[command(
    version,
    about = "Nextora native Rust inference engine with CPU and Vulkan execution"
)]
struct Cli {
    #[command(flatten)]
    engine: forge_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    forge_engine::cli::run(Cli::parse().engine)
}
