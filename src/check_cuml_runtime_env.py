#!/usr/bin/env python
"""Check whether a target environment is compatible with a local cuML wheel.

This script is intended for the "build a wheel here, install it there" workflow.
Run it on the build machine with ``--write-baseline`` and on the target machine
with ``--baseline``. It only imports heavyweight GPU packages when explicitly
asked, so metadata checks still work in partially broken environments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from typing import Any


RAPIDS_PACKAGES = [
    "cuml",
    "libcuml",
    "cudf",
    "dask-cudf",
    "dask-cuda",
    "rmm",
    "pylibraft",
    "raft-dask",
    "cuvs",
    "libcuvs",
    "libraft",
    "librmm",
    "cupy-cuda12x",
    "cuda-python",
    "numba",
    "numpy",
    "scipy",
    "scikit-learn",
    "treelite",
    "ydf",
]

RELEVANT_CONDA_PACKAGES = sorted(
    set(
        RAPIDS_PACKAGES
        + [
            "python",
            "python_abi",
            "cuda-version",
            "cuda-nvcc",
            "cuda-cudart",
            "cuda-cudart_linux-64",
            "cuda-python",
            "c-compiler",
            "cxx-compiler",
            "gcc",
            "gxx",
            "libgcc",
            "libstdcxx",
        ]
    )
)


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except FileNotFoundError:
        return {"available": False}
    except subprocess.TimeoutExpired:
        return {"available": True, "timeout": True}

    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in RAPIDS_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def conda_package_versions() -> dict[str, str]:
    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    result = run_command([conda_exe, "list", "--json"])
    if not result.get("available") or result.get("returncode") != 0:
        return {}
    try:
        packages = json.loads(result.get("stdout") or "[]")
    except json.JSONDecodeError:
        return {}
    return {
        package["name"]: package["version"]
        for package in packages
        if isinstance(package, dict)
        and isinstance(package.get("name"), str)
        and isinstance(package.get("version"), str)
    }


def parse_wheel_filename(path: Path) -> dict[str, Any]:
    """Extract wheel tags without requiring packaging.utils."""
    name = path.name
    if not name.endswith(".whl"):
        raise ValueError(f"Expected a .whl path, got {path}")

    stem = name[:-4]
    parts = stem.rsplit("-", 4)
    if len(parts) != 5:
        raise ValueError(f"Could not parse wheel filename: {name}")

    py_tag, abi_tag, platform_tag = parts[2], parts[3], parts[4]
    return {
        "filename": name,
        "distribution_version_build": "-".join(parts[:2]),
        "python_tag": py_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }


def cuda_probe(import_gpu_packages: bool) -> dict[str, Any]:
    info: dict[str, Any] = {"nvidia_smi": run_command(["nvidia-smi"])}

    if import_gpu_packages:
        try:
            import cupy as cp  # type: ignore

            info["cupy"] = {
                "imported": True,
                "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                "device_count": cp.cuda.runtime.getDeviceCount(),
            }
        except Exception as exc:  # noqa: BLE001
            info["cupy"] = {"imported": False, "error": repr(exc)}

        try:
            from numba import cuda  # type: ignore

            info["numba_cuda"] = {
                "imported": True,
                "is_available": bool(cuda.is_available()),
            }
        except Exception as exc:  # noqa: BLE001
            info["numba_cuda"] = {"imported": False, "error": repr(exc)}

    return info


def collect_environment(import_gpu_packages: bool = False, wheel: Path | None = None) -> dict[str, Any]:
    env = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "is_wsl": "microsoft" in platform.release().lower()
            or "WSL_DISTRO_NAME" in os.environ,
        },
        "conda": {
            "prefix": os.environ.get("CONDA_PREFIX"),
            "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "packages": package_versions(),
        "conda_packages": conda_package_versions(),
        "cuda": cuda_probe(import_gpu_packages),
    }
    if wheel is not None:
        env["wheel"] = parse_wheel_filename(wheel)
    return env


def baseline_environment(env: dict[str, Any]) -> dict[str, Any]:
    """Return only compatibility-relevant fields for sharing across machines."""
    cuda: dict[str, Any] = {}
    nvidia_smi = env.get("cuda", {}).get("nvidia_smi", {})
    cuda["nvidia_smi"] = {
        "available": nvidia_smi.get("available"),
        "returncode": nvidia_smi.get("returncode"),
        "timeout": nvidia_smi.get("timeout"),
    }
    if "cupy" in env.get("cuda", {}):
        cuda["cupy"] = env["cuda"]["cupy"]
    if "numba_cuda" in env.get("cuda", {}):
        cuda["numba_cuda"] = env["cuda"]["numba_cuda"]

    baseline = {
        "python": {
            "version_info": env["python"]["version_info"],
            "implementation": env["python"]["implementation"],
            "cache_tag": env["python"]["cache_tag"],
            "soabi": env["python"]["soabi"],
        },
        "platform": {
            "system": env["platform"]["system"],
            "machine": env["platform"]["machine"],
            "is_wsl": env["platform"]["is_wsl"],
        },
        "conda": {
            "default_env": env["conda"]["default_env"],
        },
        "packages": env["packages"],
        "conda_packages": {
            package: env["conda_packages"].get(package)
            for package in RELEVANT_CONDA_PACKAGES
            if env["conda_packages"].get(package) is not None
        },
        "cuda": cuda,
    }
    if "wheel" in env:
        baseline["wheel"] = env["wheel"]
    return baseline


def comparable_version(version: str | None) -> str | None:
    if version is None:
        return None
    match = re.match(r"^([0-9]+[.][0-9]+)", version)
    return match.group(1) if match else version


def wheel_python_tag_from_cache_tag(cache_tag: str | None) -> str | None:
    if cache_tag is None:
        return None
    match = re.fullmatch(r"cpython-([0-9]+)", cache_tag)
    if match:
        return f"cp{match.group(1)}"
    return cache_tag


def compare_environments(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    for key in ["version_info", "implementation", "cache_tag", "soabi"]:
        if current["python"].get(key) != baseline["python"].get(key):
            problems.append(
                f"python.{key}: current={current['python'].get(key)!r} "
                f"baseline={baseline['python'].get(key)!r}"
            )

    for key in ["system", "machine"]:
        if current["platform"].get(key) != baseline["platform"].get(key):
            problems.append(
                f"platform.{key}: current={current['platform'].get(key)!r} "
                f"baseline={baseline['platform'].get(key)!r}"
            )

    current_packages = current["packages"]
    baseline_packages = baseline["packages"]
    current_conda_packages = current.get("conda_packages", {})
    baseline_conda_packages = baseline.get("conda_packages", {})
    for package in RAPIDS_PACKAGES:
        current_raw = current_packages.get(package) or current_conda_packages.get(package)
        baseline_raw = baseline_packages.get(package) or baseline_conda_packages.get(package)
        current_version = comparable_version(current_raw)
        baseline_version = comparable_version(baseline_raw)
        if current_version != baseline_version:
            problems.append(
                f"package {package}: current={current_raw!r} "
                f"baseline={baseline_raw!r}"
            )

    wheel = baseline.get("wheel") or current.get("wheel")
    if wheel:
        cache_tag = current["python"].get("cache_tag")
        wheel_cache_tag = wheel_python_tag_from_cache_tag(cache_tag)
        soabi = current["python"].get("soabi") or ""
        platform_machine = current["platform"].get("machine")
        wheel_python_tag = wheel.get("python_tag", "")
        wheel_abi_tag = wheel.get("abi_tag", "")
        wheel_platform_tag = wheel.get("platform_tag", "")
        if wheel_cache_tag and wheel_cache_tag not in wheel_python_tag:
            problems.append(
                f"wheel python tag {wheel_python_tag!r} does not contain {wheel_cache_tag!r}"
            )
        if "cpython" in soabi and wheel_cache_tag and wheel_cache_tag not in wheel_abi_tag:
            problems.append(
                f"wheel ABI tag {wheel_abi_tag!r} does not contain {wheel_cache_tag!r}"
            )
        if platform_machine == "x86_64" and "x86_64" not in wheel_platform_tag:
            problems.append(
                f"wheel platform tag {wheel_platform_tag!r} is not linux_x86_64"
            )

    return problems


def print_summary(env: dict[str, Any], problems: list[str] | None = None) -> None:
    print("Python:")
    print(f"  executable: {env['python']['executable']}")
    print(f"  version: {'.'.join(map(str, env['python']['version_info']))}")
    print(f"  cache_tag: {env['python']['cache_tag']}")
    print(f"  SOABI: {env['python']['soabi']}")
    print("Platform:")
    print(f"  platform: {env['platform']['platform']}")
    print(f"  machine: {env['platform']['machine']}")
    print(f"  WSL: {env['platform']['is_wsl']}")
    print("Conda:")
    print(f"  env: {env['conda']['default_env']}")
    print(f"  prefix: {env['conda']['prefix']}")
    print("Key package versions:")
    for package in RAPIDS_PACKAGES:
        version = env["packages"].get(package) or env["conda_packages"].get(package)
        if version is not None:
            print(f"  {package}: {version}")
    print("CUDA:")
    nvidia_smi = env["cuda"]["nvidia_smi"]
    print(f"  nvidia-smi available: {nvidia_smi.get('available')}")
    if nvidia_smi.get("returncode") is not None:
        print(f"  nvidia-smi returncode: {nvidia_smi.get('returncode')}")
    if "cupy" in env["cuda"]:
        print(f"  cupy probe: {env['cuda']['cupy']}")
    if "numba_cuda" in env["cuda"]:
        print(f"  numba cuda probe: {env['cuda']['numba_cuda']}")
    if "wheel" in env:
        print("Wheel:")
        for key, value in env["wheel"].items():
            print(f"  {key}: {value}")

    if problems is not None:
        print("Comparison:")
        if problems:
            print("  FAIL")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("  PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check Python/CUDA/RAPIDS compatibility for installing a locally "
            "built cuML wheel on another machine."
        )
    )
    parser.add_argument("--wheel", type=Path, help="Optional cuML wheel to tag-check.")
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write the collected environment report to this JSON file.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare the current environment against a JSON baseline.",
    )
    parser.add_argument(
        "--import-gpu-packages",
        action="store_true",
        help="Also import CuPy/Numba and query CUDA. This can fail in broken GPU envs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full collected environment JSON.",
    )
    args = parser.parse_args()

    env = collect_environment(args.import_gpu_packages, args.wheel)

    problems = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        problems = compare_environments(env, baseline)

    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        portable_baseline = baseline_environment(env)
        args.write_baseline.write_text(
            json.dumps(portable_baseline, indent=2, sort_keys=True) + "\n"
        )

    if args.json:
        print(json.dumps(env, indent=2, sort_keys=True))
    else:
        print_summary(env, problems)

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
