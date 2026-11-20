use clap::Parser;

#[derive(Parser)]
#[command(
    version,
    about = "Nextora native Rust inference engine with CPU and Vulkan execution"
)]
struct Cli {
    #[command(flatten)]
    engine: nextora_engine::cli::EngineArgs,
}

fn main() -> anyhow::Result<()> {
    nextora_engine::cli::run(Cli::parse().engine)
}
