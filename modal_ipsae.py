# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Modal Labs entrypoint for running IPSAE calculations.

This wraps the DunbrackLab IPSAE CLI in a Modal container.
It accepts PAE + structure file (pdb/mmCIF), cutoff parameters,
and returns all generated outputs.

https://github.com/DunbrackLab/IPSAE

modal run ./scripts/modal_ipsae.py --input-dir ./out --models boltz --pae-cutoff 10 --dist-cutoff 10 --limit 20 --skip-existing
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import modal

HERE = Path(__file__).resolve().parent
REMOTE_REPO_PATH = Path("/workspace/IPSAE")
REMOTE_WORKDIR = Path("/tmp/ipsae")

ALLOWED_MODELS = ("af2", "af3", "boltz")


@dataclass
class IpsaeJob:
    model: str
    pae_path: Path
    struct_path: Path
    extras: Dict[str, Path]

    @property
    def label(self) -> str:
        return f"{self.model}:{self.struct_path.name}"


def _normalize_models(models: Iterable[str] | None) -> List[str]:
    """Normalize requested model identifiers to a deduplicated list."""

    if models is None:
        return list(ALLOWED_MODELS)

    tokens: List[str] = []
    if isinstance(models, str):
        tokens.extend(models.replace(",", " ").split())
    else:
        for item in models:
            if isinstance(item, str):
                tokens.extend(item.replace(",", " ").split())

    if not tokens:
        return list(ALLOWED_MODELS)

    normalized: List[str] = []
    for token in tokens:
        lowered = token.strip().lower()
        if not lowered:
            continue
        if lowered not in ALLOWED_MODELS:
            raise ValueError(
                f"Unsupported model type '{token}'. Expected one of: {', '.join(ALLOWED_MODELS)}"
            )
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized or list(ALLOWED_MODELS)


def _suffix_from_markers(stem: str, markers: Sequence[str]) -> Optional[str]:
    lowered = stem.lower()
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            return stem[idx:]
    return None


def _pick_preferred(paths: Iterable[Path], order: Sequence[str]) -> Optional[Path]:
    candidates = list(paths)
    if not candidates:
        return None
    for keyword in order:
        for candidate in candidates:
            if keyword in candidate.stem.lower():
                return candidate
    return sorted(candidates)[0]


def _discover_af2_jobs(root: Path) -> List[IpsaeJob]:
    jobs: List[IpsaeJob] = []
    json_candidates: List[tuple[str, Path]] = []
    suffix_to_structs: Dict[str, List[Path]] = {}

    for path in root.rglob("*.pdb"):
        suffix = _suffix_from_markers(path.stem, ("alphafold2", "rank_", "model_"))
        if suffix:
            suffix_to_structs.setdefault(suffix, []).append(path)

    for path in root.rglob("*.json"):
        suffix = _suffix_from_markers(path.stem, ("alphafold2", "rank_", "model_"))
        if suffix:
            json_candidates.append((suffix, path))

    seen_pairs: set[tuple[Path, Path]] = set()
    for suffix, json_path in sorted(
        json_candidates, key=lambda item: item[1].as_posix()
    ):
        struct_path = _pick_preferred(
            suffix_to_structs.get(suffix, []),
            order=("unrelaxed", "relaxed", "ranked", "model"),
        )
        if not struct_path:
            continue
        pair = (json_path.resolve(), struct_path.resolve())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        jobs.append(IpsaeJob("af2", json_path, struct_path, extras={}))

    return jobs


def _discover_af3_jobs(root: Path) -> List[IpsaeJob]:
    jobs: List[IpsaeJob] = []
    struct_lookup: Dict[tuple[str, str], Path] = {}

    for path in root.rglob("*_model_*.cif"):
        try:
            prefix, index = path.stem.rsplit("_model_", 1)
        except ValueError:
            continue
        struct_lookup[(prefix, index)] = path

    for json_path in root.rglob("*_full_data_*.json"):
        try:
            prefix, index = json_path.stem.rsplit("_full_data_", 1)
        except ValueError:
            continue
        struct_path = struct_lookup.get((prefix, index))
        if not struct_path:
            continue
        jobs.append(IpsaeJob("af3", json_path, struct_path, extras={}))

    return jobs


def _discover_boltz_jobs(root: Path) -> List[IpsaeJob]:
    jobs: List[IpsaeJob] = []

    for pae_path in sorted(root.rglob("pae_*.npz")):
        core = pae_path.stem[len("pae_") :]
        base_dir = pae_path.parent

        struct_path = base_dir / f"{core}.cif"
        confidence_path = base_dir / f"confidence_{core}.json"
        pde_path = base_dir / f"pde_{core}.npz"
        plddt_path = base_dir / f"plddt_{core}.npz"

        missing = [
            path
            for path in (struct_path, confidence_path, pde_path, plddt_path)
            if not path.is_file()
        ]

        if missing:
            # Fall back to a broader search if the files are not in the same directory.
            if not struct_path.is_file():
                struct_candidates = list(root.rglob(f"{core}.cif"))
                struct_path = struct_candidates[0] if struct_candidates else struct_path
            if not confidence_path.is_file():
                matches = list(root.rglob(f"confidence_{core}.json"))
                confidence_path = matches[0] if matches else confidence_path
            if not pde_path.is_file():
                matches = list(root.rglob(f"pde_{core}.npz"))
                pde_path = matches[0] if matches else pde_path
            if not plddt_path.is_file():
                matches = list(root.rglob(f"plddt_{core}.npz"))
                plddt_path = matches[0] if matches else plddt_path

        if not all(
            path.is_file()
            for path in (struct_path, confidence_path, pde_path, plddt_path)
        ):
            continue

        extras = {
            confidence_path.name: confidence_path,
            pde_path.name: pde_path,
            plddt_path.name: plddt_path,
        }

        jobs.append(IpsaeJob("boltz", pae_path, struct_path, extras=extras))

    return jobs


def _discover_jobs(input_dir: Path, requested_models: Sequence[str]) -> List[IpsaeJob]:
    jobs: List[IpsaeJob] = []
    missing: List[str] = []

    if "af2" in requested_models:
        af2_jobs = _discover_af2_jobs(input_dir)
        if af2_jobs:
            jobs.extend(af2_jobs)
        else:
            missing.append("af2")

    if "af3" in requested_models:
        af3_jobs = _discover_af3_jobs(input_dir)
        if af3_jobs:
            jobs.extend(af3_jobs)
        else:
            missing.append("af3")

    if "boltz" in requested_models:
        boltz_jobs = _discover_boltz_jobs(input_dir)
        if boltz_jobs:
            jobs.extend(boltz_jobs)
        else:
            missing.append("boltz")

    if missing and len(missing) == len(requested_models):
        raise FileNotFoundError(
            "No IPSAE-compatible inputs discovered for requested models: "
            + ", ".join(missing)
        )

    if missing:
        print(
            "Warning: Skipping models with no detected inputs: "
            + ", ".join(sorted(missing))
        )

    # Deduplicate identical pae/struct combinations that might be shared across models.
    unique: Dict[tuple[Path, Path], IpsaeJob] = {}
    for job in jobs:
        key = (job.pae_path.resolve(), job.struct_path.resolve())
        unique.setdefault(key, job)

    return list(unique.values())


def _job_run_metadata(job: IpsaeJob, input_root: Path) -> Tuple[str, str]:
    """Derive run identifier and model segment for output organization."""

    try:
        relative_dir = job.struct_path.parent.relative_to(input_root)
    except ValueError:
        relative_dir = Path(job.struct_path.parent.name)

    parts = relative_dir.parts

    if parts and parts[0] in ALLOWED_MODELS and len(parts) > 1:
        model_segment = parts[0]
        run_identifier = parts[1]
    elif parts:
        model_segment = job.model
        run_identifier = parts[0]
    else:
        model_segment = job.model
        run_identifier = job.struct_path.parent.stem or job.struct_path.parent.name

    if not run_identifier:
        run_identifier = job.struct_path.parent.name or "unknown"

    return run_identifier, model_segment


def execute_ipsae_jobs(
    jobs: Sequence[IpsaeJob],
    *,
    input_root: Path,
    output_root: Path,
    pae_cutoff: int,
    dist_cutoff: int,
    skip_existing: bool = False,
    limit: Optional[int] = None,
) -> int:
    """Run IPSAE for each job and write outputs under the requested root.

    Returns the number of jobs executed (skipped jobs are not counted).
    """

    executed = 0

    for job in sorted(jobs, key=lambda j: (j.model, j.struct_path.stem)):
        if not job.pae_path.is_file():
            raise FileNotFoundError(f"PAE file not found: {job.pae_path}")
        if not job.struct_path.is_file():
            raise FileNotFoundError(f"Structure file not found: {job.struct_path}")

        run_identifier, model_segment = _job_run_metadata(job, input_root)
        job_out_dir = output_root / run_identifier
        if model_segment:
            job_out_dir /= model_segment

        if skip_existing and job_out_dir.exists():
            print(
                f"Skipping run {run_identifier} ({job.label}) - outputs exist at {job_out_dir}"
            )
            continue

        job_out_dir.mkdir(parents=True, exist_ok=True)

        extra_files = {
            name: path.read_bytes()
            for name, path in job.extras.items()
            if path.is_file()
        }

        print(f"Processing run {run_identifier} ({job.label})")
        results = run_ipsae.remote(
            pae_bytes=job.pae_path.read_bytes(),
            pae_name=job.pae_path.name,
            struct_bytes=job.struct_path.read_bytes(),
            struct_name=job.struct_path.name,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
            extra_files=extra_files or None,
        )

        for name, blob in results.get("files", {}).items():
            (job_out_dir / (job.struct_path.stem + name)).write_bytes(blob)

        if results.get("stdout"):
            print(results["stdout"], end="")
        if results.get("stderr"):
            print(results["stderr"], end="", file=os.sys.stderr)

        executed += 1
        if limit is not None and executed >= limit:
            break

    return executed


def _gpu_from_env() -> Optional[modal.gpu.GpuType]:
    gpu_env = os.environ.get("GPU", "").upper()
    if gpu_env == "A10G":
        return modal.gpu.A10G()
    if gpu_env == "A100":
        return modal.gpu.A100()
    if gpu_env == "H100":
        return modal.gpu.H100()
    if gpu_env == "L4":
        return modal.gpu.L4()
    return None


GPU_CONFIG = _gpu_from_env()
TIMEOUT_MINUTES = int(os.environ.get("TIMEOUT", "60"))

# Build image: clone IPSAE and install requirements
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/DunbrackLab/IPSAE {REMOTE_REPO_PATH}")
    .pip_install("numpy")
)

app = modal.App("ipsae")


def _prepare_workdir() -> Path:
    if REMOTE_WORKDIR.exists():
        shutil.rmtree(REMOTE_WORKDIR)
    REMOTE_WORKDIR.mkdir(parents=True, exist_ok=True)
    return REMOTE_WORKDIR


def _collect_outputs(
    structure_path: Path, pae_cutoff: int, dist_cutoff: int
) -> Dict[str, bytes]:
    """Collect IPSAE output files next to structure file."""
    stem = structure_path.with_suffix("").as_posix()
    stem = f"{stem}_{pae_cutoff:02d}_{dist_cutoff:02d}"

    outputs: Dict[str, bytes] = {}
    for suffix in [".txt", "_byres.txt", ".pml"]:
        path = Path(f"{stem}{suffix}")
        if path.exists():
            outputs[suffix] = path.read_bytes()
    return outputs


@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT_MINUTES * 60)
def run_ipsae(
    *,
    pae_bytes: bytes,
    pae_name: str,
    struct_bytes: bytes,
    struct_name: str,
    pae_cutoff: int,
    dist_cutoff: int,
    extra_files: Dict[str, bytes] | None = None,
) -> Dict[str, object]:
    """Remote execution wrapper for IPSAE CLI."""
    workdir = _prepare_workdir()

    pae_path = workdir / pae_name
    struct_path = workdir / struct_name
    pae_path.write_bytes(pae_bytes)
    struct_path.write_bytes(struct_bytes)

    for name, blob in (extra_files or {}).items():
        (workdir / name).write_bytes(blob)

    cmd: List[str] = [
        "python",
        str(REMOTE_REPO_PATH / "ipsae.py"),
        str(pae_path),
        str(struct_path),
        str(pae_cutoff),
        str(dist_cutoff),
    ]

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            f"IPSAE failed ({completed.returncode})\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    outputs = _collect_outputs(struct_path, pae_cutoff, dist_cutoff)

    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "files": outputs,
    }


@app.local_entrypoint()
def main(
    input_dir: str,
    models: str | None = None,
    pae_cutoff: int = 10,
    dist_cutoff: int = 10,
    run_name: str | None = None,
    out_dir: str = "./out/ipsae",
    skip_existing: bool = False,
    limit: int | None = None,
) -> None:
    """Discover IPSAE inputs for the requested models and run them on Modal."""

    input_root = Path(input_dir)
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    requested_models = _normalize_models(models)
    jobs = _discover_jobs(input_root, requested_models)
    if not jobs:
        raise RuntimeError("No IPSAE jobs detected. Please verify the input directory.")

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    base_out_dir = Path(out_dir)
    if run_name:
        base_out_dir /= run_name
    base_out_dir.mkdir(parents=True, exist_ok=True)

    executed = execute_ipsae_jobs(
        jobs,
        input_root=input_root,
        output_root=base_out_dir,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        skip_existing=skip_existing,
        limit=limit,
    )

    print(f"IPSAE outputs written to {base_out_dir} ({executed} job(s) run)")
