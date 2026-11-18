fn main() -> anyhow::Result<()> {
    rusting_strata::cli::run_from(std::env::args_os())
}
