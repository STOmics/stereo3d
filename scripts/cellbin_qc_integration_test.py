import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


DEFAULT_H5AD_NAMES = [
    "D01153A4.h5ad",
    "D01153A5.h5ad",
    "D01153A6.h5ad",
    "D01153B4.h5ad",
    "A00792A2.h5ad",
]


def matrix_stats(adata: AnnData):
    x = adata.X
    if sparse.issparse(x):
        total_counts = np.asarray(x.sum(axis=1)).ravel()
        n_genes = np.asarray((x > 0).sum(axis=1)).ravel()
        nnz = x.nnz
    else:
        arr = np.asarray(x)
        total_counts = arr.sum(axis=1)
        n_genes = (arr > 0).sum(axis=1)
        nnz = int((arr > 0).sum())
    sparsity = 1 - nnz / (adata.n_obs * adata.n_vars)
    return total_counts, n_genes, sparsity


def qc_row(sample, stage, adata, total_counts, n_genes, sparsity):
    return {
        "sample": sample,
        "stage": stage,
        "n_cells": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "median_counts": float(np.median(total_counts)),
        "mean_counts": float(np.mean(total_counts)),
        "p05_counts": float(np.percentile(total_counts, 5)),
        "p95_counts": float(np.percentile(total_counts, 95)),
        "median_detected_genes": float(np.median(n_genes)),
        "mean_detected_genes": float(np.mean(n_genes)),
        "p05_detected_genes": float(np.percentile(n_genes, 5)),
        "p95_detected_genes": float(np.percentile(n_genes, 95)),
        "sparsity": float(sparsity),
    }


def load_and_filter_h5ads(
    h5ad_dir: Path,
    names,
    min_counts: float,
    min_genes: int,
    max_counts_quantile: float,
    max_genes_quantile: float,
    max_cells_per_slice: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    adatas = []
    qc_rows = []

    for name in names:
        path = h5ad_dir / name
        if not path.exists():
            raise FileNotFoundError(path)

        sample = path.stem
        adata = sc.read_h5ad(path)
        adata.var_names_make_unique()

        total_counts, n_genes, sparsity = matrix_stats(adata)
        qc_rows.append(qc_row(sample, "before", adata, total_counts, n_genes, sparsity))

        keep = np.ones(adata.n_obs, dtype=bool)
        if min_counts > 0:
            keep &= total_counts >= min_counts
        if min_genes > 0:
            keep &= n_genes >= min_genes
        if 0 < max_counts_quantile < 1:
            keep &= total_counts <= np.quantile(total_counts, max_counts_quantile)
        if 0 < max_genes_quantile < 1:
            keep &= n_genes <= np.quantile(n_genes, max_genes_quantile)

        idx = np.flatnonzero(keep)
        if max_cells_per_slice > 0 and idx.size > max_cells_per_slice:
            idx = rng.choice(idx, size=max_cells_per_slice, replace=False)
            idx = np.sort(idx)

        adata = adata[idx].copy()
        adata.obs["sample"] = sample
        adata.obs["batch"] = sample
        adata.obs_names = [f"{sample}_{x}" for x in adata.obs_names.astype(str)]

        total_counts, n_genes, sparsity = matrix_stats(adata)
        adata.obs["qc_total_counts"] = total_counts
        adata.obs["qc_n_genes"] = n_genes
        qc_rows.append(qc_row(sample, "after", adata, total_counts, n_genes, sparsity))
        adatas.append(adata)

    return adatas, pd.DataFrame(qc_rows)


def summarize_crosstab(crosstab: pd.DataFrame) -> pd.DataFrame:
    totals = crosstab.sum(axis=0)
    props = crosstab.div(totals, axis=1).fillna(0)
    return pd.DataFrame(
        {
            "cluster": totals.index.astype(str),
            "total": totals.to_numpy(dtype=int),
            "dominant_batch": props.idxmax(axis=0).astype(str).to_numpy(),
            "max_batch_prop": props.max(axis=0).to_numpy(),
            "active_batches": (crosstab > 0).sum(axis=0).to_numpy(dtype=int),
        }
    ).sort_values(["total"], ascending=False)


def print_crosstab_metrics(crosstab: pd.DataFrame):
    total = crosstab.sum(axis=0)
    dom = crosstab.max(axis=0) / total.replace(0, pd.NA)
    print(f"cluster total = {len(total)}")
    for threshold in [100, 1000, 5000]:
        s = dom[total >= threshold]
        print(f"\ncluster cells >= {threshold} n = {len(s)}")
        for frac in [0.7, 0.8, 0.9, 0.95, 0.99]:
            print(f"  dominant >= {frac}: {int((s >= frac).sum())}/{len(s)}")


def run_integration(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = args.names or DEFAULT_H5AD_NAMES
    adatas, qc = load_and_filter_h5ads(
        Path(args.h5ad_dir),
        names,
        args.min_counts,
        args.min_genes,
        args.max_counts_quantile,
        args.max_genes_quantile,
        args.max_cells_per_slice,
        args.seed,
    )
    qc.to_csv(outdir / "qc_summary.csv", index=False)
    qc.to_excel(outdir / "qc_summary.xlsx", index=False)

    adata = sc.concat(adatas, join="inner", label="concat_batch", keys=[a.obs["sample"].iloc[0] for a in adatas])
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.write_h5ad(outdir / "00_qc_filtered_concat_raw.h5ad")
    print(f"concat shape = {adata.shape}")

    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(args.n_hvg, adata.n_vars),
        batch_key="batch",
        flavor="seurat",
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < 2:
        raise ValueError(f"Not enough HVGs: {n_hvg}")

    n_comps = min(args.n_pcs, adata.n_obs - 1, n_hvg - 1)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", use_highly_variable=True)

    rep = "X_pca"
    if not args.no_harmony:
        sc.external.pp.harmony_integrate(
            adata,
            key="batch",
            theta=args.theta,
            max_iter_harmony=args.max_iter_harmony,
        )
        rep = "X_pca_harmony"

    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep=rep)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=args.resolution)

    crosstab = pd.crosstab(adata.obs["batch"], adata.obs["leiden"])
    crosstab.to_csv(outdir / "slice_cluster_crosstab.csv")
    crosstab.to_excel(outdir / "slice_cluster_crosstab.xlsx")

    summary = summarize_crosstab(crosstab)
    summary.to_csv(outdir / "cluster_batch_summary.csv", index=False)
    summary.to_excel(outdir / "cluster_batch_summary.xlsx", index=False)

    adata.write_h5ad(outdir / "01_integrated_clustered.h5ad")

    import matplotlib.pyplot as plt

    sc.pl.umap(adata, color=["batch", "leiden"], show=False)
    plt.savefig(outdir / "umap_batch_leiden.png", bbox_inches="tight", dpi=180)
    plt.close("all")

    for sample in adata.obs["batch"].cat.categories:
        sub = adata[adata.obs["batch"] == sample].copy()
        if "spatial" in sub.obsm:
            sc.pl.spatial(sub, color="leiden", spot_size=args.spot_size, show=False)
            plt.savefig(outdir / f"{sample}_leiden.png", bbox_inches="tight", dpi=180)
            plt.close("all")

    print("\nQC summary:")
    print(qc.to_string(index=False))
    print("\nCrosstab metrics:")
    print_crosstab_metrics(crosstab)
    print(f"\nDONE: {outdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Non-invasive cellbin QC/filter/integration test. Reads H5ADs and writes a separate test outdir."
    )
    parser.add_argument("--h5ad-dir", required=True, help="Directory containing raw cellbin H5ADs.")
    parser.add_argument("--outdir", required=True, help="Separate output directory for this test run.")
    parser.add_argument("--names", nargs="*", default=None, help="Ordered H5AD file names. Defaults to mouse 5-slice names.")
    parser.add_argument("--min-counts", type=float, default=150)
    parser.add_argument("--min-genes", type=int, default=80)
    parser.add_argument("--max-counts-quantile", type=float, default=0.0, help="0 disables upper count filtering.")
    parser.add_argument("--max-genes-quantile", type=float, default=0.0, help="0 disables upper gene filtering.")
    parser.add_argument("--max-cells-per-slice", type=int, default=0, help="0 disables downsampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--theta", type=float, default=4.0)
    parser.add_argument("--max-iter-harmony", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--resolution", type=float, default=0.4)
    parser.add_argument("--spot-size", type=float, default=15)
    parser.add_argument("--no-harmony", action="store_true")
    args = parser.parse_args()
    run_integration(args)


if __name__ == "__main__":
    main()
