import argparse
import os
import sys
from pathlib import Path

import numpy as np

from run_mouse_5slice import (
    ensure_path,
    load_config,
    prepare_matrix_dir,
    project_root,
    validate_expected_files,
)


def add_project_to_path() -> Path:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def get_ordered_mask_paths(mask_dir: Path, chip_seq: list) -> list:
    paths = []
    missing = []
    for chip in chip_seq:
        path = mask_dir / f"{chip}.tif"
        if path.exists():
            paths.append(str(path))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing mask files:\n" + "\n".join(missing))
    return paths


def cluster_sort_key(value: str):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def get_ordered_matrix_paths(matrix_dir: Path, chip_seq: list) -> list:
    paths = []
    missing = []
    for chip in chip_seq:
        candidates = [
            matrix_dir / f"{chip}.gef",
            matrix_dir / f"{chip}.gem.gz",
            matrix_dir / f"{chip}.gem",
            matrix_dir / f"{chip}.txt",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            missing.append(f"{chip}.gef/.gem.gz/.gem/.txt under {matrix_dir}")
        else:
            paths.append(str(found))
    if missing:
        raise FileNotFoundError("Missing matrix files:\n" + "\n".join(missing))
    return paths


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def should_run_dir(path: Path, expected_count: int, overwrite: bool, suffix: str = "*") -> bool:
    if overwrite:
        return True
    if not path.exists():
        return True
    return len(list(path.glob(suffix))) < expected_count


def get_z_index_list(slice_seq) -> list:
    if hasattr(slice_seq, "z_index_list"):
        return list(slice_seq.z_index_list)
    if hasattr(slice_seq, "get_z_interval"):
        return list(slice_seq.get_z_interval(index="short").values())
    raise AttributeError("SliceSequence does not expose z_index_list or get_z_interval().")


def transform_matrix(matrix_paths: list, crop_json: Path, align_json: Path, output_dir: Path, overwrite: bool) -> None:
    from stereo3d.gem.transform import trans_matrix_by_json

    ensure_dirs(output_dir)
    if overwrite or not any(output_dir.iterdir()):
        trans_matrix_by_json(matrix_paths, str(crop_json), str(align_json), str(output_dir))
    else:
        print(f"Skip existing transformed matrix: {output_dir}")


def create_outer_mesh(mask_paths: list, slice_seq, output_dir: Path, overwrite: bool) -> None:
    from stereo3d.mesh.create_mesh_3d import get_mask_3d_points, points_3d_to_mesh

    ensure_dirs(output_dir)
    mesh_obj = output_dir / "mask_mesh.obj"
    if mesh_obj.exists() and not overwrite:
        print(f"Skip existing outer mesh: {mesh_obj}")
        return

    points_3d = get_mask_3d_points(
        mask_paths,
        get_z_index_list(slice_seq),
        z_interval=slice_seq.z_interval,
        pixel4mm=slice_seq.size_per_pixel,
        output_path=str(output_dir),
    )
    points_3d_to_mesh(
        points_3d,
        z_interval=slice_seq.z_interval,
        mesh_scale=1,
        output_path=str(output_dir),
        show_mesh=False,
    )


def h5ad_names_from_matrix_paths(matrix_paths: list) -> list:
    names = []
    for path in matrix_paths:
        chip = Path(path).name.split(".")[0]
        names.append(f"{chip}.h5ad")
    return names


def run_cellbin_leiden(
    cellbin_matrix_dir: Path,
    tissue_matrix_dir: Path,
    cellbin_matrix_paths: list,
    slice_seq,
    output_path: Path,
    overwrite: bool,
) -> None:
    from stereo3d.h5ad.txt2adata import batch_cluster, batch_spatial_leiden
    from stereo3d.h5ad.uniform_cluster_color_v2 import (
        organ_mesh,
        read_and_parse_by_celltype,
        uniform_cluster_color,
    )
    import tqdm

    transform_h5ad = output_path / "05.transform_cellbin"
    color_h5ad = output_path / "06.color_cellbin"
    organ = output_path / "07.organ_cellbin"
    ensure_dirs(transform_h5ad, color_h5ad, organ)

    expected_h5ad = h5ad_names_from_matrix_paths(cellbin_matrix_paths)

    if should_run_dir(transform_h5ad, len(expected_h5ad), overwrite, "*.h5ad"):
        print(f"Create cellbin H5AD: {transform_h5ad}")
        batch_cluster(
            matrix_dir=str(cellbin_matrix_dir),
            save_dir=str(transform_h5ad),
            gene_name_dir=str(tissue_matrix_dir),
        )
    else:
        print(f"Skip existing cellbin H5AD: {transform_h5ad}")

    if overwrite or len(list(transform_h5ad.glob("*.png"))) < len(expected_h5ad):
        batch_spatial_leiden(h5ad_path=str(transform_h5ad), save_path=str(transform_h5ad))
    else:
        print(f"Skip existing per-slice Leiden images: {transform_h5ad}")

    h5ad_list = [str(transform_h5ad / name) for name in expected_h5ad]
    missing_h5ad = [path for path in h5ad_list if not Path(path).exists()]
    if missing_h5ad:
        raise FileNotFoundError("Missing generated H5AD files:\n" + "\n".join(missing_h5ad))

    if should_run_dir(color_h5ad, len(expected_h5ad), overwrite, "*.h5ad"):
        categories = uniform_cluster_color(h5ad_list, str(color_h5ad), z_index_list=get_z_index_list(slice_seq))
    else:
        print(f"Skip existing cross-slice colored H5AD: {color_h5ad}")
        first = next(color_h5ad.glob("*.h5ad"))
        import scanpy as sc

        adata = sc.read_h5ad(first)
        categories = sorted(map(str, adata.obs["leiden"].cat.categories), key=cluster_sort_key)

    print(f"Cellbin Leiden categories: {categories}")
    color_h5ad_list = [str(color_h5ad / name) for name in expected_h5ad]

    existing_organs = list(organ.glob("*.obj"))
    if existing_organs and not overwrite:
        print(f"Skip existing cellbin organ meshes: {organ}")
        return

    for c in tqdm.tqdm(categories, desc="Cellbin Organ", ncols=100):
        organ_path = read_and_parse_by_celltype(
            outdir=str(organ),
            spatial_regis="spatial_mm",
            anno="leiden",
            celltype=c,
            adata_list=None,
            h5ad_list=color_h5ad_list,
            sc_xyz=None,
            z_index_list=get_z_index_list(slice_seq),
        )
        try:
            organ_mesh(organ_path, organ_path.replace(".txt", ".obj"), z_interval=slice_seq.z_interval)
        except Exception as exc:
            print(f"Organ {c} failed: {exc}")


def run_combined(config: dict) -> None:
    add_project_to_path()

    from stereo3d.file.slice import SliceSequence
    from stereo3d.register.mask_crop import cut_mask
    from stereo3d.register.registration import align_slices

    output_path = Path(config["output_path"])
    ensure_dirs(output_path)

    tissue_matrix_path = ensure_path(config["tissue_matrix_path"], "tissue_matrix_path")
    cellbin_matrix_path = ensure_path(config["cellbin_matrix_path"], "cellbin_matrix_path")
    tissue_mask = ensure_path(config["tissue_mask"], "tissue_mask")
    record_sheet = ensure_path(config["record_sheet"], "record_sheet")

    prepared_cellbin = prepare_matrix_dir(
        matrix_path=cellbin_matrix_path,
        output_path=output_path,
        matrix_file_mode=config.get("cellbin_matrix_file_mode", "cellbin_suffix"),
    )
    validate_expected_files(tissue_matrix_path, tissue_mask, record_sheet)
    validate_expected_files(prepared_cellbin, tissue_mask, record_sheet)

    numba_cache_dir = Path(config.get("numba_cache_dir", output_path / ".numba_cache"))
    ensure_dirs(numba_cache_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

    overwrite = bool(config.get("overwriter", 1))
    registration = bool(config.get("registration", 1))

    slice_seq = SliceSequence()
    slice_seq.from_xlsx(str(record_sheet))
    chip_seq = list(slice_seq.get_chip_seq())

    mask_paths = get_ordered_mask_paths(tissue_mask, chip_seq)
    tissue_matrix_paths = get_ordered_matrix_paths(tissue_matrix_path, chip_seq)
    cellbin_matrix_paths = get_ordered_matrix_paths(prepared_cellbin, chip_seq)

    crop_mask_dir = output_path / "02.register" / "00.crop_mask"
    align_mask_dir = output_path / "02.register" / "01.align_mask"
    ensure_dirs(crop_mask_dir, align_mask_dir)

    if should_run_dir(crop_mask_dir, len(mask_paths), overwrite, "*.tif"):
        cut_mask(mask_paths, str(crop_mask_dir), registration)
    else:
        print(f"Skip existing crop masks: {crop_mask_dir}")

    crop_mask_paths = [str(crop_mask_dir / Path(path).name) for path in mask_paths]
    if should_run_dir(align_mask_dir, len(mask_paths), overwrite, "*.tif"):
        align_slices(crop_mask_paths, str(align_mask_dir), registration=registration)
    else:
        print(f"Skip existing aligned masks: {align_mask_dir}")

    crop_json = crop_mask_dir / "mask_cut_info.json"
    align_json = align_mask_dir / "align_info.json"
    if not crop_json.exists() or not align_json.exists():
        raise FileNotFoundError(f"Missing registration jsons: {crop_json}, {align_json}")

    transform_matrix(
        tissue_matrix_paths,
        crop_json,
        align_json,
        output_path / "03.tissue_matrix",
        overwrite=overwrite,
    )
    tissue_transformed = output_path / "03.tissue_matrix"
    cellbin_transformed = output_path / "03.cellbin_matrix"
    transform_matrix(
        cellbin_matrix_paths,
        crop_json,
        align_json,
        cellbin_transformed,
        overwrite=overwrite,
    )

    aligned_mask_paths = [str(align_mask_dir / Path(path).name) for path in mask_paths]
    create_outer_mesh(aligned_mask_paths, slice_seq, output_path / "04.mesh", overwrite=overwrite)

    transformed_cellbin_paths = get_ordered_matrix_paths(cellbin_transformed, chip_seq)
    run_cellbin_leiden(
        cellbin_matrix_dir=cellbin_transformed,
        tissue_matrix_dir=tissue_transformed,
        cellbin_matrix_paths=transformed_cellbin_paths,
        slice_seq=slice_seq,
        output_path=output_path,
        overwrite=overwrite,
    )

    print("\nCombined output check:")
    for name in [
        "02.register",
        "03.tissue_matrix",
        "03.cellbin_matrix",
        "04.mesh",
        "05.transform_cellbin",
        "06.color_cellbin",
        "07.organ_cellbin",
    ]:
        path = output_path / name
        print(f"  {name}: {'OK' if path.exists() else 'MISSING'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a combined Stereo3D workflow: mask/tissue for geometry, cellbin for Leiden/organ annotation."
    )
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "mouse_embryo_5slice_combined.json"),
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_combined(config)


if __name__ == "__main__":
    main()
