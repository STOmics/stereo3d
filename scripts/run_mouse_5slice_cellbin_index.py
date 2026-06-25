import argparse
import os
import sys
from pathlib import Path

from run_mouse_5slice import (
    ensure_path,
    load_config,
    prepare_matrix_dir,
    project_root,
    validate_expected_files,
)
from run_mouse_5slice_combined import (
    add_project_to_path,
    cluster_sort_key,
    create_outer_mesh,
    ensure_dirs,
    get_ordered_mask_paths,
    get_ordered_matrix_paths,
    get_z_index_list,
    h5ad_names_from_matrix_paths,
    should_run_dir,
    transform_matrix,
)


def run_cellbin_index_leiden(
    cellbin_matrix_dir: Path,
    cellbin_matrix_paths: list,
    slice_seq,
    output_path: Path,
    overwrite: bool,
    run_organ: bool,
) -> None:
    from stereo3d.h5ad.txt2adata import batch_cluster, batch_spatial_leiden
    from stereo3d.h5ad.uniform_cluster_color_v2 import (
        organ_mesh,
        read_and_parse_by_celltype,
        uniform_cluster_color,
    )
    import scanpy as sc
    import tqdm

    transform_h5ad = output_path / "05.transform_cellbin"
    color_h5ad = output_path / "06.color_cellbin"
    organ = output_path / "07.organ_cellbin"
    ensure_dirs(transform_h5ad, color_h5ad, organ)

    expected_h5ad = h5ad_names_from_matrix_paths(cellbin_matrix_paths)

    if should_run_dir(transform_h5ad, len(expected_h5ad), overwrite, "*.h5ad"):
        print(f"Create cellbin-index H5AD: {transform_h5ad}")
        batch_cluster(
            matrix_dir=str(cellbin_matrix_dir),
            save_dir=str(transform_h5ad),
        )
    else:
        print(f"Skip existing cellbin-index H5AD: {transform_h5ad}")

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
        adata = sc.read_h5ad(first)
        categories = sorted(map(str, adata.obs["leiden"].cat.categories), key=cluster_sort_key)

    print(f"Cellbin-index Leiden categories: {categories}")
    if not run_organ:
        print(f"Skip cellbin-index organ meshes because run_organ is false. Color output is ready: {color_h5ad}")
        return

    color_h5ad_list = [str(color_h5ad / name) for name in expected_h5ad]

    existing_organs = list(organ.glob("*.obj"))
    if existing_organs and not overwrite:
        print(f"Skip existing cellbin-index organ meshes: {organ}")
        return

    for c in tqdm.tqdm(categories, desc="Cellbin-index Organ", ncols=100):
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


def run_cellbin_index(config: dict) -> None:
    add_project_to_path()

    from stereo3d.file.slice import SliceSequence
    from stereo3d.register.mask_crop import cut_mask
    from stereo3d.register.registration import align_slices

    output_path = Path(config["output_path"])
    ensure_dirs(output_path)

    cellbin_matrix_path = ensure_path(config["cellbin_matrix_path"], "cellbin_matrix_path")
    mask_path = ensure_path(config["mask_path"], "mask_path")
    record_sheet = ensure_path(config["record_sheet"], "record_sheet")

    prepared_cellbin = prepare_matrix_dir(
        matrix_path=cellbin_matrix_path,
        output_path=output_path,
        matrix_file_mode=config.get("cellbin_matrix_file_mode", "cellbin_suffix"),
    )
    validate_expected_files(prepared_cellbin, mask_path, record_sheet)

    numba_cache_dir = Path(config.get("numba_cache_dir", output_path / ".numba_cache"))
    ensure_dirs(numba_cache_dir)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

    overwrite = bool(config.get("overwriter", 1))
    registration = bool(config.get("registration", 1))

    slice_seq = SliceSequence()
    slice_seq.from_xlsx(str(record_sheet))
    chip_seq = list(slice_seq.get_chip_seq())

    mask_paths = get_ordered_mask_paths(mask_path, chip_seq)
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
    run_cellbin_index_leiden(
        cellbin_matrix_dir=cellbin_transformed,
        cellbin_matrix_paths=transformed_cellbin_paths,
        slice_seq=slice_seq,
        output_path=output_path,
        overwrite=overwrite,
        run_organ=bool(config.get("run_organ", 1)),
    )

    print("\nCellbin-index output check:")
    for name in [
        "02.register",
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
        description=(
            "Run a cellbin-index Stereo3D workflow. Cellbin integer geneID "
            "is used as the cross-slice gene key."
        )
    )
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs" / "mouse_embryo_5slice_cellbin_index.json"),
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    run_cellbin_index(config)


if __name__ == "__main__":
    main()
