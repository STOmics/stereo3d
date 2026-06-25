import os
import json
import shutil
import gzip

import numpy as np
import pandas as pd
import h5py

from glob import glob
from tqdm import tqdm
from stereo3d.manual.align_parameters import apply_affine_deformation


def trans_points(x, y, offset=None, mat=None, map_x=None, map_y=None):
    """
    Args:
        x:
        y:
        offset: [int, int] - Here offset is the starting coordinate of cutting
        mat:
    Returns:
        coord: x, y
    """
    coord = np.array([x, y])
    coord = coord.transpose(1, 0)

    if offset:
        coord[:, 0] = coord[:, 0] - offset[0]
        coord[:, 1] = coord[:, 1] - offset[1]

    if mat and len(mat) > 3:
        mat = mat[2:]

    if mat:
        coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis=1)
        cor_trans_val = np.transpose(coord)

        cor_pro = np.dot(mat, cor_trans_val)

        x_arr = cor_pro[0, :].transpose()
        y_arr = cor_pro[1, :].transpose()

        coord = np.concatenate([np.expand_dims(x_arr, axis=1),
                                np.expand_dims(y_arr, axis=1)], axis=1)

    if map_x is not None and map_y is not None:
        x_temp = np.clip(coord[:, 0].astype(int), 0, map_x.shape[1] - 1)
        y_temp = np.clip(coord[:, 1].astype(int), 0, map_y.shape[0] - 1)
        coord = np.column_stack([map_x[y_temp, x_temp], map_y[y_temp, x_temp]])

    return coord[:, 0], coord[:, 1]


def gem_read(
        gem_file
):
    """
    Args:
        gem_file:
    """
    suffix = os.path.splitext(gem_file)[1]
    if suffix == ".gz":
        fh = gzip.open(gem_file, "rb")
    else:
        fh = open(str(gem_file), "rb")  # pylint: disable=consider-using-with
    title = ""
    # Move pointer to the header of line
    eoh = 0
    header = ""
    for line in fh:
        line = line.decode("utf-8")
        if not line.startswith("#"):
            title = line
            break
        header += line
        eoh = fh.tell()
    fh.seek(eoh)
    # Initlise
    # title = title.strip("\n").split("\t")
    title = title.strip().split("\t")
    umi_count_name = [i for i in title if "ount" in i][0]
    title = ["x", "y", umi_count_name]

    type_dict = {"geneID": str}
    type_dict.update(dict(zip(title, [np.uint32] * len(title))))
    title.insert(0, "geneID")

    df = pd.read_csv(
        fh,
        sep="\t",
        header=0,
        usecols=title,
        dtype=type_dict,
    )

    return df


def read_cellbin_from_gef(gef_file, gene_name_gef=None):
    with h5py.File(gef_file, "r") as h:
        if "cellBin" not in h:
            raise ValueError(f"Not a cellbin gef: {gef_file}")

        g = h["cellBin"]
        cell = g["cell"][:]
        gene = g["gene"][:]
        gene_exp = g["geneExp"][:] if "geneExp" in g else None
        cell_exp = g["cellExp"][:] if "cellExp" in g else None

    if cell.shape[0] == 0:
        return pd.DataFrame(columns=("geneID", "x", "y", "MIDCount", "cellID"))

    gene_name_field = "geneID" if "geneID" in gene.dtype.names else "geneName"
    gene_names, mapped_gene_count = _cellbin_gene_names(gene, gene_name_field, gene_name_gef)

    if gene_exp is not None and "cellCount" in gene.dtype.names:
        gene_cell_count = gene["cellCount"].astype(np.int64)
        total_nnz = int(gene_cell_count.sum())
        if total_nnz != int(gene_exp.shape[0]):
            raise ValueError(
                f"cellbin nnz mismatch: sum(gene.cellCount)={total_nnz}, geneExp={gene_exp.shape[0]}"
            )

        offsets = gene["offset"].astype(np.int64) if "offset" in gene.dtype.names else np.cumsum(
            np.r_[0, gene_cell_count[:-1]]
        )
        expected_offsets = np.cumsum(np.r_[0, gene_cell_count[:-1]])
        if np.array_equal(offsets, expected_offsets):
            exp = gene_exp
            gene_idx = np.repeat(np.arange(gene.shape[0], dtype=np.int64), gene_cell_count)
        else:
            nonzero_gene = gene_cell_count > 0
            exp_index = np.concatenate([
                np.arange(offset, offset + count, dtype=np.int64)
                for offset, count in zip(offsets, gene_cell_count)
                if count > 0
            ])
            exp = gene_exp[exp_index]
            gene_idx = np.repeat(np.arange(gene.shape[0], dtype=np.int64)[nonzero_gene], gene_cell_count[nonzero_gene])

        if mapped_gene_count < gene.shape[0]:
            keep = gene_idx < mapped_gene_count
            exp = exp[keep]
            gene_idx = gene_idx[keep]

        if exp.shape[0] == 0:
            return pd.DataFrame(columns=("geneID", "x", "y", "MIDCount", "cellID"))

        cell_ref = exp["cellID"].astype(np.int64)
        if cell_ref.size and cell_ref.max() < cell.shape[0]:
            cell_row = cell_ref
            cell_id = cell["id"].astype(np.int64)[cell_row]
        else:
            cell_index = pd.Index(cell["id"].astype(np.int64))
            cell_row = cell_index.get_indexer(cell_ref)
            if (cell_row < 0).any():
                raise ValueError("cellbin geneExp references unknown cell ids")
            cell_id = cell_ref

        return pd.DataFrame({
            "geneID": gene_names[gene_idx],
            "x": cell["x"].astype(np.int64)[cell_row],
            "y": cell["y"].astype(np.int64)[cell_row],
            "MIDCount": exp["count"].astype(np.int64),
            "cellID": cell_id,
        })

    if cell_exp is None:
        return pd.DataFrame(columns=("geneID", "x", "y", "MIDCount", "cellID"))

    cell_gene_count = cell["geneCount"].astype(np.int64)
    total_nnz = int(cell_gene_count.sum())
    if total_nnz != int(cell_exp.shape[0]):
        raise ValueError(
            f"cellbin nnz mismatch: sum(cell.geneCount)={total_nnz}, cellExp={cell_exp.shape[0]}"
        )

    offsets = cell["offset"].astype(np.int64) if "offset" in cell.dtype.names else np.cumsum(
        np.r_[0, cell_gene_count[:-1]]
    )
    expected_offsets = np.cumsum(np.r_[0, cell_gene_count[:-1]])
    if np.array_equal(offsets, expected_offsets):
        exp = cell_exp
        cell_row = np.repeat(np.arange(cell.shape[0], dtype=np.int64), cell_gene_count)
    else:
        nonzero_cell = cell_gene_count > 0
        exp_index = np.concatenate([
            np.arange(offset, offset + count, dtype=np.int64)
            for offset, count in zip(offsets, cell_gene_count)
            if count > 0
        ])
        exp = cell_exp[exp_index]
        cell_row = np.repeat(np.arange(cell.shape[0], dtype=np.int64)[nonzero_cell], cell_gene_count[nonzero_cell])

    gene_idx = exp["geneID"].astype(np.int64)
    if mapped_gene_count < gene.shape[0]:
        keep = gene_idx < mapped_gene_count
        exp = exp[keep]
        gene_idx = gene_idx[keep]
        cell_row = cell_row[keep]

    if exp.shape[0] == 0:
        return pd.DataFrame(columns=("geneID", "x", "y", "MIDCount", "cellID"))

    df = pd.DataFrame({
        "geneID": gene_names[gene_idx],
        "x": cell["x"].astype(np.int64)[cell_row],
        "y": cell["y"].astype(np.int64)[cell_row],
        "MIDCount": exp["count"].astype(np.int64),
        "cellID": cell["id"].astype(np.int64)[cell_row],
    })
    return df


def read_gene_names_from_gef(gef_file):
    with h5py.File(gef_file, "r") as h:
        gene = h["geneExp"]["bin1"]["gene"][:]
    gene_name_field = "geneID" if "geneID" in gene.dtype.names else "gene"
    return gene[gene_name_field].astype("U")


def _cellbin_gene_names(gene, gene_name_field, gene_name_gef=None):
    gene_names = gene[gene_name_field].astype("U")
    mapped_gene_count = gene_names.shape[0]
    if gene_name_gef is not None and os.path.exists(gene_name_gef):
        ref_names = read_gene_names_from_gef(gene_name_gef)
        mapped_gene_count = min(ref_names.shape[0], gene_names.shape[0])
        gene_names = gene_names[:mapped_gene_count]
        if mapped_gene_count > 0:
            gene_names[:mapped_gene_count] = ref_names[:mapped_gene_count]

    safe_names = []
    seen = set()
    for i, raw_name in enumerate(gene_names):
        name = str(raw_name).strip()
        if not name or name in seen:
            name = f"gene_{i}"
        seen.add(name)
        safe_names.append(name)
    return np.array(safe_names, dtype=object), mapped_gene_count


def is_cellbin_gef(gef_file):
    with h5py.File(gef_file, "r") as h:
        return "cellBin" in h


def read_gem_from_gef(gef_file):
    h = h5py.File(gef_file, 'r')
    gene = h['geneExp']['bin1']['gene'][:]
    expression = h['geneExp']['bin1']['expression'][:]
    df = pd.DataFrame(columns=('geneID', 'x', 'y', 'MIDCount'))
    df['x'] = expression['x']
    df['y'] = expression['y']
    df['MIDCount'] = expression['count']
    _ = np.zeros((expression.shape[0],), dtype='S64')

    gene_name = 'geneID' if 'geneID' in gene.dtype.names else 'gene'
    for i in range(gene.shape[0]):
        s, o = (gene[i]['offset'], gene[i]['count'])
        df['geneID'][s: s + o] = gene[i][gene_name]

    return df


def _deformation_map(mat, shape):
    if mat and len(mat) > 3 and shape is not None:
        return apply_affine_deformation(shape[0], shape[1], mat[0], mat[1], alpha=1.0)
    return None, None


def gef_trans(gef_file, offset, mat, shape, output_path):
    shutil.copy(gef_file, output_path)
    map_x, map_y = _deformation_map(mat, shape)
    with h5py.File(output_path, 'r+') as h:
        if 'cellBin' in h:
            cell_ds = h['cellBin']['cell']
            cell = cell_ds[:]
            new_x, new_y = trans_points(cell['x'], cell['y'], offset, mat, map_x, map_y)
            cell['x'] = np.int32(np.round(new_x))
            cell['y'] = np.int32(np.round(new_y))
            cell_ds[:] = cell

            if 'cellBorder' in h['cellBin']:
                border_ds = h['cellBin']['cellBorder']
                border = border_ds[:]
                flat_x = border[:, :, 0].reshape(-1)
                flat_y = border[:, :, 1].reshape(-1)
                border_x, border_y = trans_points(flat_x, flat_y, offset, mat, map_x, map_y)
                border[:, :, 0] = np.int16(np.round(border_x).reshape(border.shape[0], border.shape[1]))
                border[:, :, 1] = np.int16(np.round(border_y).reshape(border.shape[0], border.shape[1]))
                border_ds[:] = border
        else:
            exp_ds = h['geneExp']['bin1']['expression']
            expression = exp_ds[:]
            new_x, new_y = trans_points(expression['x'], expression['y'], offset, mat, map_x, map_y)
            expression['x'] = new_x
            expression['y'] = new_y
            exp_ds[:] = expression


def anndata_trans(adata_file, offset, mat, shape, output_path):
    import scanpy as sc
    adata = sc.read_h5ad(adata_file)
    map_x, map_y = _deformation_map(mat, shape)
    if "spatial" in adata.obsm.keys():
        x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
        new_x, new_y = trans_points(x, y, offset, mat, map_x, map_y)
        adata.obsm["spatial"][:, 0] = new_x
        adata.obsm["spatial"][:, 1] = new_y
        adata.write_h5ad(output_path)


def gem_trans(gem_file, offset, mat, shape, output_path):
    """
    Args:
        gem_file:
        offset:
        mat:
        output_path: str - With file name
    """
    gem = gem_read(gem_file)
    map_x, map_y = _deformation_map(mat, shape)

    # gem['x'] = gem['x'] - min(gem['x'])
    # gem['y'] = gem['y'] - min(gem['y'])

    new_x, new_y = trans_points(gem['x'], gem['y'], offset, mat, map_x, map_y)

    gem['x'] = np.int_(np.round(new_x))
    gem['y'] = np.int_(np.round(new_y))
    gem.to_csv(output_path, sep='\t', index=False)


def trans_matrix_by_json(gem_path, cut_json_path, align_json_path, output_path):
    """
    Args:
        gem_path:
        cut_json_path:
        align_json_path:
        output_path:
    """
    if isinstance(gem_path, str):
        gem_list = glob(os.path.join(gem_path, '*.*'))
    elif isinstance(gem_path, list):
        gem_list = gem_path

    os.makedirs(output_path, exist_ok=True)
    with open(cut_json_path, 'r') as js:
        mask_cut_info = json.load(js)
    with open(align_json_path, 'r') as js:
        align_info = json.load(js)

    for matrix_file in tqdm(gem_list, desc='Gem', ncols=100):
        matrix_name = os.path.basename(matrix_file).split('.')[0]
        for key in mask_cut_info.keys():
            if matrix_name in key:
                mask_cut = mask_cut_info[key]
                break
        else:
            mask_cut = None

        for key in align_info.keys():
            if matrix_name in key:
                align = align_info[key]
                break
        else:
            align = None

        if mask_cut is not None or align is not None:
            # mask_cut = None
            mat = align['mat'] if align is not None else None
            shape = align.get('shape') if align is not None else None
            if matrix_file.endswith('txt') or matrix_file.endswith('gem') or matrix_file.endswith('gem.gz'):
                gem_trans(
                    matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.gem")
                )
            elif matrix_file.endswith('gef'):
                gef_trans(
                    matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.gef")
                )
            elif matrix_file.endswith('.h5ad'):
                anndata_trans(matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.h5ad"))


if __name__ == "__main__":
    # trans_matrix_by_json(gem_path=r"D:\02.data\E14-16h_a_bin1_image_gem",
    #                   cut_json_path=r"D:\02.data\E14-16h_a_bin1_image_regis\align_info.json",
    #                   align_json_path=r"D:\02.data\E14-16h_a_bin1_image_regis\align_info.json",
    #                   output_path=r"D:\02.data\E14-16h_a_bin1_image_gem\new_gem")
    # read_gem_from_gef(r"/media/Data1/user/szl/data/output/gem/SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gef")

    aaa = gem_read(r"D:\02.data\SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gem.gz")
