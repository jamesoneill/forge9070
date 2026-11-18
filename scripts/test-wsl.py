#!/usr/bin/env python3
"""Test launcher arguments with fake executables; does not claim WSL/GPU execution."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Launchers(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="forge9070-wsl-test-")
        self.base = Path(self.directory.name)
        self.log = self.base / "calls.jsonl"
        self.bin = self.base / "fake engine"
        self.bin.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['FORGE_TEST_LOG'], 'a') as f:
    f.write(json.dumps({'args':sys.argv[1:], 'wslenv':os.environ.get('WSLENV','')})+'\\n')
if sys.argv[1:2] == ['doctor']:
    sys.exit(int(os.environ.get('FORGE_TEST_DOCTOR_EXIT','0')))
""")
        self.bin.chmod(0o755)
        self.env = os.environ.copy()
        self.env["FORGE_TEST_LOG"] = str(self.log)

    def tearDown(self):
        self.directory.cleanup()

    def run_script(self, script, *args):
        return subprocess.run(["bash", str(ROOT / "scripts" / script), *args],
                              env=self.env, text=True, capture_output=True, timeout=10)

    def calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_cpu_paths_remain_single_arguments(self):
        models = self.base / "models with spaces; not a command"
        model = self.base / "model with spaces.gguf"
        model.touch()
        r = self.run_script("serve-wsl.sh", "--backend", "cpu", "--bin", str(self.bin),
                            "--models-dir", str(models), "--model", str(model),
                            "--", "--ctx", "1024")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls()[0]["args"], ["serve", "--models-dir", str(models),
                         "--port", "8080", "--model", str(model), "--cpu", "--ctx", "1024"])

    def test_native_gpu_failure_never_launches_or_falls_back(self):
        self.env["FORGE_TEST_DOCTOR_EXIT"] = "7"
        r = self.run_script("serve-wsl.sh", "--backend", "vulkan", "--bin", str(self.bin),
                            "--models-dir", str(self.base / "models"))
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual([c["args"] for c in self.calls()], [["doctor", "--require-gpu"]])

    def test_model_directory_env_matches_cli_and_explicit_flag_wins(self):
        configured = self.base / "configured model catalogue"
        self.env["FORGE9070_MODELS_DIR"] = str(configured)
        r = self.run_script("serve-wsl.sh", "--backend", "cpu", "--bin", str(self.bin))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls()[0]["args"][1:3], ["--models-dir", str(configured)])
        override = self.base / "explicit catalogue"
        r = self.run_script("serve-wsl.sh", "--backend", "cpu", "--bin", str(self.bin),
                            "--models-dir", str(override))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.calls()[1]["args"][1:3], ["--models-dir", str(override)])

    def test_native_gpu_success_launches_after_doctor(self):
        r = self.run_script("serve-wsl.sh", "--backend", "vulkan", "--bin", str(self.bin),
                            "--models-dir", str(self.base / "models"))
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls()
        self.assertEqual(calls[0]["args"], ["doctor", "--require-gpu"])
        self.assertEqual(calls[1]["args"][0], "serve")
        self.assertNotIn("--cpu", calls[1]["args"])

    def test_backend_and_port_are_validated(self):
        r = self.run_script("serve-wsl.sh", "--bin", str(self.bin))
        self.assertEqual(r.returncode, 2)
        r = self.run_script("serve-wsl.sh", "--backend", "cpu", "--bin", str(self.bin), "--port", "99999")
        self.assertEqual(r.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_windows_paths_and_env_are_forwarded_without_shell_evaluation(self):
        exe = self.base / "Windows engine.exe"
        exe.write_bytes(self.bin.read_bytes())
        exe.chmod(0o755)
        wslpath = self.base / "wslpath"
        wslpath.write_text("#!/usr/bin/env python3\nimport sys\nassert sys.argv[1]=='-w'\nprint('C:'+sys.argv[2].replace('/', '\\\\'))\n")
        wslpath.chmod(0o755)
        self.env["PATH"] = str(self.base) + os.pathsep + self.env["PATH"]
        self.env["HF_TOKEN"] = "fake-test-token"
        self.env["STRATA_API_KEY"] = "fake-test-key"
        self.env["WSLENV"] = "KEEP/p:HF_TOKEN/u:STRATA_API_KEY/u"
        models = self.base / "Models with spaces"
        model = self.base / "chosen model.gguf"
        model.touch()
        r = self.run_script("serve-wsl-windows.sh", "--exe", str(exe),
                            "--models-dir", str(models), "--model", str(model), "--", "--ctx", "4096")
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls()
        self.assertEqual(calls[0]["args"], ["doctor", "--require-gpu"])
        self.assertEqual(calls[1]["args"], ["serve", "--models-dir", "C:"+str(models).replace("/", "\\"),
                         "--port", "8080", "--model", "C:"+str(model).replace("/", "\\"), "--ctx", "4096"])
        self.assertEqual(set(calls[1]["wslenv"].split(":")), {"KEEP/p", "HF_TOKEN/w", "STRATA_API_KEY/w"})
        self.assertNotIn("fake-test-token", r.stdout + r.stderr)
        self.assertNotIn("fake-test-key", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
