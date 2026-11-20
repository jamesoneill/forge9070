# Start Nextora on Windows

This is the supported **Windows-native** route for the Nextora Rust inference engine. It does **not** require WSL, ROCm, Python, Docker or a downloaded model merely to open the configuration UI.

## 1. Prerequisites

- Windows 10/11 x64 and PowerShell or PowerShell 7.
- Git, [Rustup](https://rustup.rs/) and the **MSVC** Rust target, with the Rust version pinned by `rust-toolchain.toml`.
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/): choose **Desktop development with C++**, including the MSVC compiler and Windows SDK.
- For *GPU inference*, a vendor driver providing a compatible hardware Vulkan adapter. The UI and CPU backend can work even without one.

Run these in **Windows PowerShell**, not an Ubuntu/WSL shell:

```powershell
rustup show active-toolchain
cargo --version
git --version
```

## 2. Clone, build and open the UI

Use a local Windows drive, not `\\wsl.localhost\...`:

```powershell
New-Item -ItemType Directory -Force C:\code | Out-Null
Set-Location C:\code
git clone https://github.com/jamesoneill/forge9070.git nextora
Set-Location .\nextora
.\scripts\configure-native-windows.ps1 -Build
```

The launcher builds the `nextora-engine` **Cargo package** and starts
`target\release\nextora.exe` (the executable is named `nextora.exe`,
**not** `nextora-engine.exe`). It opens <http://127.0.0.1:9070/>.
If your browser fails to open automatically, open that address manually.
**Keep the terminal running**; this is a foreground server, not an installed
Windows service.

The first start does not need `--model` or `-ModelsDir`. Choose a directory
and select **Scan models** in the UI only after the server is running. To use a
known directory explicitly:

```powershell
.\scripts\configure-native-windows.ps1 -ModelsDir 'F:\Models'
```

Change that path to your actual folder. An empty or missing `F:\Models` is not
a valid example on every computer.

## 3. Verify the executable independently

From the repository root:

```powershell
Test-Path .\target\release\nextora.exe
.\target\release\nextora.exe --help
.\target\release\nextora.exe doctor
.\target\release\nextora.exe capabilities
```

`doctor` reports device eligibility; it is **not** a performance or physical-GPU
execution test. If only CPU or unsuitable Vulkan devices appear, start with:

```powershell
.\scripts\configure-native-windows.ps1 -Backend cpu
```

`-Backend auto` selects a compatible Vulkan adapter when available and otherwise
selects CPU during discovery. It will not silently switch to CPU after a model,
shader or GPU allocation error. For troubleshooting, explicit CPU is useful.

To launch without the PowerShell helper:

```powershell
.\target\release\nextora.exe serve --backend auto --open-ui --settings-file "$env:LOCALAPPDATA\Nextora\native-service.json"
```

Make sure the settings-file parent directory exists before using a custom
direct command. The helper creates it automatically.

## Common startup errors

| Symptom | Cause / next step |
| --- | --- |
| `nextora.exe` is not recognised | Use `.\target\release\nextora.exe` in the repository root; do not type `nextora-engine.exe`. |
| Executable missing | Run `.\scripts\configure-native-windows.ps1 -Build`; inspect the Cargo error if compilation fails. |
| `cargo` not recognised | Install Rustup and reopen PowerShell; check `cargo --version`. |
| `link.exe` or MSVC toolchain errors | Install Visual Studio C++ Build Tools and Windows SDK. Use native Windows MSVC, not a WSL Linux build. |
| Windows Application Control blocks Rust build scripts | Build on a Windows-local path. If still blocked, your device's policy administrator must allow the signed/toolchain executables; the launcher does not disable security policy. |
| Browser opens but cannot connect | Keep the server terminal open; read errors and check whether port 9070 is already in use. |
| `Address already in use` | Close the other server, or pass `-Port 9071` and open `http://127.0.0.1:9071/`. |
| Saved settings unexpectedly choose a model/GPU | Existing saved profiles override model/runtime flags. Use `-SettingsFile "$env:TEMP\nextora-test-profile.json"` for a non-destructive fresh profile. |
| Model isn't found or refuses to load | Start without a model; select the real model folder and read the scan rejection. Nextora requires supported complete GGUF shard sets and compatible tokenizer metadata. |
| GPU not detected | Check `nextora.exe doctor` and update the vendor driver if appropriate. ROCm/DKMS is not the inference backend; Vulkan support is required. |
| UI asks for an API key | You started the server with `--api-key`; enter that key. The normal launcher does not configure one. |
| Scanning `F:\` is slow or truncated | Select the smallest known folder of complete GGUF shards, e.g. `F:\Models\MyModel`. |

You can safely test with a fresh profile without deleting your current settings:

```powershell
.\scripts\configure-native-windows.ps1 -Build -Backend cpu `
    -SettingsFile "$env:TEMP\nextora-troubleshoot.json" -Port 9071
```

This preserves your existing Nextora profile and makes it clear whether the
problem is device/model configuration or the basic UI startup.

## Working with WSL

If you want the GPU exposed **through Windows**, build `nextora.exe` with
Windows Rust/MSVC and launch it from Windows PowerShell as above. Clients
inside WSL can connect to the Windows service using your actual WSL
network configuration. The native WSL Linux binary is a separate build and
requires a compatible Vulkan adapter in the Linux environment; `/dev/dxg`
or `llvmpipe` alone doesn't establish that.

See the [native engine guide](NATIVE-ENGINE.md) and [model-loading
guide](MODEL-LOADING.md) for supported architectures, memory budgets, GGUF
validation, API and performance measurements.
