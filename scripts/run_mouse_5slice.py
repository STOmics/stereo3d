import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_path(path: str, label: str) -> Path:
    value = Path(path)
    if not value.exists():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_matrix_dir(matrix_path: Path, output_path: Path, matrix_file_mode: str) -> Path:
    """Return a matrix directory whose file names match Stereo3D chip lookup rules."""
    if matrix_file_mode == "standard":
        return matrix_path
    if matrix_file_mode != "cellbin_suffix":
        raise ValueError(f"Unsupported matrix_file_mode: {matrix_file_mode}")

    prepared = output_path / "_prepared_inputs" / "cellbin_matrix"
    prepared.mkdir(parents=True, exist_ok=True)
    for src in matrix_path.glob("*.cellbin.gef"):
        chip = src.name.replace(".cellbin.gef", "")
        dst = prepared / f"{chip}.gef"
        link_or_copy(src, dst)
    if not any(prepared.glob("*.gef")):
        raise FileNotFoundError(f"No *.cellbin.gef files found in {matrix_path}")
    return prepared


def validate_expected_files(matrix_dir: Path, mask_dir: Path, record_xlsx: Path) -> None:
    slice_df = pd.read_excel(record_xlsx, sheet_name="SliceSequence")
    missing = []
    for chip in slice_df["SSDNA_ChipNo"].astype(str).tolist():
        mask = mask_dir / f"{chip}.tif"
        candidates = [
            matrix_dir / f"{chip}.gef",
            matrix_dir / f"{chip}.gem.gz",
            matrix_dir / f"{chip}.gem",
            matrix_dir / f"{chip}.txt",
        ]
        if not mask.exists():
            missing.append(str(mask))
        if not any(p.exists() for p in candidates):
            missing.append(f"{chip}.gef/.gem.gz/.gem/.txt under {matrix_dir}")
    if missing:
        raise FileNotFoundError("Missing expected input files:\n" + "\n".join(missing))


def run_stereo3d(config: dict) -> None:
    root = project_root()
    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    matrix_path = ensure_path(config["matrix_path"], "matrix_path")
    tissue_mask = ensure_path(config["tissue_mask"], "tissue_mask")
    record_sheet = ensure_path(config["record_sheet"], "record_sheet")

    prepared_matrix = prepare_matrix_dir(
        matrix_path=matrix_path,
        output_path=output_path,
        matrix_file_mode=config.get("matrix_file_mode", "standard"),
    )
    validate_expected_files(prepared_matrix, tissue_mask, record_sheet)

    numba_cache_dir = Path(config.get("numba_cache_dir", output_path / ".numba_cache"))
    numba_cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

    stereo3d_script = root / "stereo3d" / "stereo3d_with_matrix.py"
    cmd = [
        sys.executable,
        str(stereo3d_script),
        "--matrix_path",
        str(prepared_matrix),
        "--tissue_mask",
        str(tissue_mask),
        "--record_sheet",
        str(record_sheet),
        "--output",
        str(output_path),
        "--registration",
        str(config.get("registration", 1)),
        "--overwriter",
        str(config.get("overwriter", 1)),
    ]

    print("Running Stereo3D command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, cwd=str(root), env=env, check=True)

    expected_dirs = ["02.register", "03.matrix", "04.mesh", "05.transform", "06.color", "07.organ"]
    print("\nOutput check:")
    for name in expected_dirs:
        path = output_path / name
        print(f"  {name}: {'OK' if path.exists() else 'MISSING'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stereo3D on the mouse embryo 5-slice dataset.")
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "mouse_embryo_5slice_tissuegef.json"),
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_stereo3d(config)


if __name__ == "__main__":
    main()
