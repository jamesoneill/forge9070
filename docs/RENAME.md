# Nextora migration and compatibility

**Nextora** is the public identity for the independent Rust inference runtime.
The primary executable is `nextora`; its Rust library package is `nextora-engine`.
The auxiliary `nextora-cli` forwards to the same runtime. No alternative
provider runtime or additional inference route is installed.

## Install from source

```sh
git clone https://github.com/jamesoneill/forge9070.git nextora
cd nextora
cargo build --release --locked -p nextora-engine
./target/release/nextora doctor
```

The repository URL is retained because this connection cannot perform a GitHub
repository rename. Rename the GitHub repository to `nextora` with the owner
account before switching links; redirects should be verified first.

Existing settings profiles, model files and GGUF shard names are not
renamed automatically. The HTTP interface and native port 9070 are unchanged.
The native runtime no longer provides the retired provider/downloader interface
or the port 8080 service. Consult [current capabilities](NATIVE-ENGINE.md)
and [validation evidence](../VALIDATION.md) before migrating clients.

Historical release records and third-party attribution remain intact for
traceability. A name change does not certify universal GPU compatibility or
model quality.
