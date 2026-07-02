#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cellbin.gef -> AnnData (cell-level)

cell x gene sparse matrix 

Gene names: use geneName if filled (safe for multi-slice inner-join by name);
if empty, fall back to placeholder gene_{i} and mark var_names_source
"""
import os

import numpy as np
import h5py
import scipy.sparse as sp
import pandas as pd
from anndata import AnnData


def _decode_gene_names(gene):
    """Decode /cellBin/gene geneName -> (names, all_empty)."""
    raw = np.array([g['geneName'] for g in gene])  # S64
    names = np.array([b.split(b'\x00')[0].decode(errors='replace').strip() for b in raw])
    all_empty = bool((names == '').all())
    return names, all_empty


def _make_unique(names):
    """Ensure var_names non-empty and unique: fill blanks with gene_{i}, dedup dups."""
    out = np.array([(n if n else f'gene_{i}') for i, n in enumerate(names)])
    if len(set(out)) != len(out):
        seen = {}
        uniq = []
        for nm in out:
            if nm in seen:
                seen[nm] += 1
                uniq.append(f'{nm}_dup{seen[nm]}')
            else:
                seen[nm] = 0
                uniq.append(nm)
        out = np.array(uniq)
    return out


def read_cellbin_gef(gef_file):
    """Read cellbin.gef into a cell-level AnnData (no bin aggregation).

    X=CSR(count); obs has cell meta; obsm['spatial']=[x,y]; obsm['cell_border']
    if present; uns marks bin_type/data_unit/var_names_source.
    """
    h = h5py.File(gef_file, 'r')

    cell = h['cellBin']['cell'][:]
    gene = h['cellBin']['gene'][:]
    cell_exp = h['cellBin']['cellExp'][:]

    n_cell = cell.shape[0]
    n_gene = gene.shape[0]

    # var_names: prefer geneName, else placeholder gene_{i}
    gene_names, all_empty = _decode_gene_names(gene)
    if all_empty:
        var_names = np.array([f'gene_{i}' for i in range(n_gene)])
        var_names_source = 'index_placeholder'
    else:
        var_names = _make_unique(gene_names)
        var_names_source = 'geneName' if (gene_names != '').all() else 'geneName_with_placeholder_fallback'

    # Build cell x gene CSR. cellExp is cell-major with contiguous offsets, so
    # cellExp[:total] holds all entries; row index via repeat(geneCount).
    gene_count = cell['geneCount'].astype(np.int64)
    total = int(gene_count.sum())
    if total > cell_exp.shape[0]:
        # Defensive: non-contiguous offsets -> fall back to per-cell concat
        import glog
        glog.warning(
            f'geneCount.sum()={total} > len(cellExp)={cell_exp.shape[0]}, '
            f'fallback to per-cell segment concatenation.'
        )
        rows, cols, vals = [], [], []
        for ci in range(n_cell):
            s = int(cell[ci]['offset'])
            e = s + int(cell[ci]['geneCount'])
            seg = cell_exp[s:e]
            rows.append(np.full(seg.shape[0], ci, dtype=np.int64))
            cols.append(seg['geneID'].astype(np.int64))
            vals.append(seg['count'].astype(np.float32))
        row = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        col = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        data = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
    else:
        row = np.repeat(np.arange(n_cell, dtype=np.int64), gene_count)
        col = cell_exp['geneID'][:total].astype(np.int64)
        data = cell_exp['count'][:total].astype(np.float32)

    if col.size and col.max() >= n_gene:
        raise ValueError(
            f'cellExp geneID max {col.max()} >= n_gene {n_gene}, '
            f'gene table inconsistent with cellExp.'
        )

    X = sp.csr_matrix((data, (row, col)), shape=(n_cell, n_gene))

    obs = pd.DataFrame({
        'cell_id': cell['id'].astype(np.int64),
        'x': cell['x'].astype(np.int64),
        'y': cell['y'].astype(np.int64),
        'geneCount': cell['geneCount'].astype(np.int64),
        'expCount': cell['expCount'].astype(np.int64),
        'dnbCount': cell['dnbCount'].astype(np.int64),
        'area': cell['area'].astype(np.int64),
        'cellTypeID': cell['cellTypeID'].astype(np.int64),
        'clusterID': cell['clusterID'].astype(np.int64),
    }, index=cell['id'].astype(str))
    obs.index.name = 'cell_id'

    var = pd.DataFrame({
        'cellCount': gene['cellCount'].astype(np.int64),
        'expCount': gene['expCount'].astype(np.int64),
        'maxMIDcount': gene['maxMIDcount'].astype(np.int64),
    }, index=var_names)

    adata = AnnData(X=X, obs=obs, var=var)

    adata.obsm['spatial'] = np.column_stack([cell['x'].astype(np.float64),
                                             cell['y'].astype(np.float64)])
    if 'cellBorder' in h['cellBin']:
        adata.obsm['cell_border'] = h['cellBin']['cellBorder'][:].astype(np.int64)

    # uns: lets downstream (mesh binsize, var_names guard) recognize cellbin
    adata.uns['bin_type'] = 'cellbin'
    adata.uns['data_unit'] = {'binsize': 'cellbin'}
    adata.uns['var_names_source'] = var_names_source

    h.close()
    return adata


def cellbin_to_h5ad(gef_file, save_path):
    """Read cellbin.gef and write .h5ad."""
    adata = read_cellbin_gef(gef_file)
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    adata.write_h5ad(save_path)
    return save_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='cellbin.gef -> h5ad (cell-level)')
    parser.add_argument('-i', '--input', required=True, help='.cellbin.gef file')
    parser.add_argument('-o', '--output', required=True, help='output .h5ad file')
    args = parser.parse_args()
    cellbin_to_h5ad(args.input, args.output)
    print(f'[cellbin2adata] {args.input} -> {args.output} done')
