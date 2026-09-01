from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
WRAPPER = REPOSITORY / "scripts" / "run_with_chatbot_libtorch.sh"
CACHE_HEADER = "# This is the CMakeCache file."


class RunWithChatbotLibtorchTests(unittest.TestCase):
    def make_layout(
        self,
        root: Path,
        *,
        torch_entries: list[str] | None = None,
        chatbot_entry: str = "SNNBASE_EXPERIMENTS_ENABLE_CHATBOT:BOOL=ON",
        cache_header: str = CACHE_HEADER,
        create_torch_config: bool = True,
        create_libtorch: bool = True,
    ) -> tuple[Path, Path, Path]:
        torch_root = root / "Torch Prefix=Pinned"
        torch_dir = torch_root / "share" / "cmake" / "Torch"
        torch_library_dir = torch_root / "lib"
        torch_dir.mkdir(parents=True)
        torch_library_dir.mkdir(parents=True)
        if create_torch_config:
            (torch_dir / "TorchConfig.cmake").write_text(
                "# fixture\n", encoding="utf-8"
            )
        if create_libtorch:
            (torch_library_dir / "libtorch.so").write_bytes(b"fixture")

        build_dir = root / "Build Dir"
        build_dir.mkdir()
        if torch_entries is None:
            torch_entries = [f"Torch_DIR:PATH={torch_dir}"]
        cache_lines = [cache_header, chatbot_entry, *torch_entries]
        (build_dir / "CMakeCache.txt").write_text(
            "\n".join(cache_lines) + "\n", encoding="utf-8"
        )
        return build_dir, torch_dir, torch_library_dir

    @staticmethod
    def run_wrapper(
        arguments: list[str], *, environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(WRAPPER), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_exec_preserves_argv_and_prepends_only_configured_torch_lib(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_dir, _, torch_library_dir = self.make_layout(root)
            sentinel = root / "shell-was-evaluated"
            payload = [
                "space separated",
                f"; touch {sentinel}",
                f"$(touch {sentinel})",
                f"`touch {sentinel}`",
                "*.json",
                "--build-dir",
                "command-owned-value",
                "--",
            ]
            probe = textwrap.dedent("""
                import json
                import os
                import subprocess
                import sys

                child = subprocess.check_output(
                    [sys.executable, "-c", "import os; print(os.environ['LD_LIBRARY_PATH'])"],
                    text=True,
                ).strip()
                print(json.dumps({
                    "argv": sys.argv[1:],
                    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
                    "child_ld_library_path": child,
                }))
                """)
            environment = os.environ.copy()
            environment["LD_LIBRARY_PATH"] = "/existing/one:/existing path/two"
            result = self.run_wrapper(
                [
                    "--build-dir",
                    str(build_dir),
                    "--",
                    sys.executable,
                    "-c",
                    probe,
                    *payload,
                ],
                environment=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            expected_ld = (
                f"{torch_library_dir.resolve()}:" "/existing/one:/existing path/two"
            )
            self.assertEqual(observed["argv"], payload)
            self.assertEqual(observed["ld_library_path"], expected_ld)
            self.assertEqual(observed["child_ld_library_path"], expected_ld)
            self.assertFalse(sentinel.exists())

    def test_unset_library_path_becomes_exact_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_dir, _, torch_library_dir = self.make_layout(root)
            environment = os.environ.copy()
            environment.pop("LD_LIBRARY_PATH", None)
            result = self.run_wrapper(
                [
                    "--build-dir",
                    str(build_dir),
                    "--",
                    sys.executable,
                    "-c",
                    "import os; print(os.environ['LD_LIBRARY_PATH'])",
                ],
                environment=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(torch_library_dir.resolve()))

    def test_missing_separator_command_and_conflicting_options_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_dir, _, _ = self.make_layout(root)
            cases = [
                [],
                ["--build-dir"],
                ["--build-dir", str(build_dir)],
                ["--build-dir", str(build_dir), "--"],
                ["--", "/bin/true"],
                ["--build-dir", str(build_dir), "/bin/true"],
                [
                    "--build-dir",
                    str(build_dir),
                    "--build-dir",
                    str(root / "Other Build"),
                    "--",
                    "/bin/true",
                ],
            ]
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    result = self.run_wrapper(arguments)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Usage:", result.stderr)

    def test_missing_and_malformed_caches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = self.run_wrapper(
                ["--build-dir", str(root / "missing"), "--", "/bin/true"]
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn("not a configured CMake build directory", missing.stderr)

            malformed_dir, _, _ = self.make_layout(
                root / "malformed", cache_header="# not a cache"
            )
            malformed = self.run_wrapper(
                ["--build-dir", str(malformed_dir), "--", "/bin/true"]
            )
            self.assertEqual(malformed.returncode, 1)
            self.assertIn("malformed CMake cache header", malformed.stderr)

    def test_cache_rejects_missing_duplicate_and_non_path_torch_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_torch_dir = (
                root / "reference" / "Torch Prefix=Pinned" / "share" / "cmake" / "Torch"
            )
            cases = {
                "missing": ([], "does not record Torch_DIR"),
                "duplicate": (
                    [
                        f"Torch_DIR:PATH={valid_torch_dir}",
                        f"Torch_DIR:PATH={valid_torch_dir}",
                    ],
                    "conflicting Torch_DIR entries",
                ),
                "wrong-type": (
                    [f"Torch_DIR:STRING={valid_torch_dir}"],
                    "malformed Torch_DIR entry",
                ),
                "relative": (
                    ["Torch_DIR:PATH=relative/torch"],
                    "must be an absolute path",
                ),
            }
            for name, (entries, expected_error) in cases.items():
                with self.subTest(name=name):
                    build_dir, _, _ = self.make_layout(
                        root / name, torch_entries=entries
                    )
                    result = self.run_wrapper(
                        ["--build-dir", str(build_dir), "--", "/bin/true"]
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(expected_error, result.stderr)

    def test_cache_requires_chatbot_config_and_real_torch_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chatbot_off, _, _ = self.make_layout(
                root / "chatbot-off",
                chatbot_entry="SNNBASE_EXPERIMENTS_ENABLE_CHATBOT:BOOL=OFF",
            )
            no_config, _, _ = self.make_layout(
                root / "no-config", create_torch_config=False
            )
            no_library, _, _ = self.make_layout(
                root / "no-library", create_libtorch=False
            )
            cases = [
                (chatbot_off, "not configured with chatbot support"),
                (no_config, "missing TorchConfig.cmake"),
                (no_library, "missing libtorch.so"),
            ]
            for build_dir, expected_error in cases:
                with self.subTest(build_dir=build_dir):
                    result = self.run_wrapper(
                        ["--build-dir", str(build_dir), "--", "/bin/true"]
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(expected_error, result.stderr)


if __name__ == "__main__":
    unittest.main()
