import os
from plyfile import PlyData, PlyElement
import pandas as pd
import anndata as ad
import numpy as np
import argparse
import colorcet as cc
import matplotlib.colors as mcolors
import random
from scipy.sparse import issparse


def classify_rows(row):  
    """
    Screening of co-expressing cells
    """
    non_zero_columns = row.index[row.ne(0)]  
    return '#'.join(non_zero_columns) 


def normalize_array(array,low_percent,high_percent):
    """
    Normalizes an array by specifying lower and upper bounds
    """
    min_val = np.min(array)
    max_val = np.max(array)
    norm_array = (array - min_val) / (max_val - min_val)
    q1 = np.percentile(array, low_percent)
    q3 = np.percentile(array, high_percent)
    mask = (array >= q1) & (array <= q3)
    norm_array[mask] = (array[mask] - q1) / (q3 - q1)
    norm_array[~mask] = 0
    return norm_array


def value_to_color(array, crange ,cmap):
    """
    Convert the values in a one-dimensional array to color RGB values.

    parameter：
    value_array: An array containing the values to be converted to colors.
    cmap_name: String specifying the name of the colormap to use, defaults to 'rainbow'.

    return： An array containing the RGB color values corresponding to each value.

    """

    low, high = crange
    # Extract the color list of the colormap
    
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    # Normalize the array to be between 0 and 1
    norm_array = normalize_array(array,low,high)

    # Get the index corresponding to the normalized value
    index_array = np.floor(norm_array * (len(color_list)-1)).astype(int)

    # Get the color string corresponding to the normalized value
    colors = [color_list[i] for i in index_array]

    # Convert hexadecimal color to RGB color
    rgb_array = np.array([np.array([int(color[i:i+2], 16)
                         for i in (1, 3, 5)]) for color in colors])

    # Returns the corresponding color value
    return rgb_array


def color_mix(colora,colorb):
    """ Mixing two colors """
    mixcolor = []
    rgba = np.array([int(colora[i:i+2], 16) for i in (1, 3, 5)]) 
    rgbb = np.array([int(colorb[i:i+2], 16) for i in (1, 3, 5)]) 
    
    for i in range(3):
        colorTop = rgba[i]/255
        colorBottom = rgbb[i]/255
        newColor = max(colorTop, colorBottom)
        # if(colorTop == 0):
        #     newColor = 0
        # else:
        #     newColor = 1 - min(1, (1 - colorBottom) / colorTop)
        mixcolor.append(newColor)
    result = tuple(int(x*255) for x in mixcolor) 
    r, g, b = result
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def convert_to_mm(df, resolution):
    """ Convert coordinates to mm """
    df['x'] *= resolution[0]
    df['y'] *= resolution[1]
    df['z'] *= resolution[2]

    df['x'] /= 1000000
    df['y'] /= 1000000
    df['z'] /= 1000000

    return df


def add_zrandom(df, zdistance):
    """
    Add a random offset to the z value of a single piece of data. The offset range is half of the minimum spacing between adjacent pieces.
    """
    random.seed(0)
    zdistance /= 1000000
    df['z'] = df['z'].apply(
        lambda x: x + random.uniform(-zdistance/2, zdistance/2))

    return df


def make_celltype_ply_data(inputdata, outputdir, prefix,  position, label, colordict, resolution, zdistance, withvalue, withid):
    """
    The coordinates and classification results are converted into point cloud files, and the classification results are
    color mapped according to the specified color mapping table.

    parameter:
    inputdata: String, a list text file storing all single-slice h5ad and corresponding z values after multiplexing
               and registration.
    outputdir: A string that specifies the output file path.
    prefix: A string specifying the output file prefix.
    position: String, used to read coordinates in h5ad.
    label: String, used to read annotation results in h5ad.
    colordict: A string that specifies where to read the colormap.
    resolution: Integer array, saves the actual physical length of each axial coordinate unit 1 in the unified
                coordinate system of all slices, in nm.
    zdistance: Integer, saves the minimum value of the distance between all adjacent slices.
    withvalue: Boolean, whether to save annotation category information.
    withid: Boolean, indicates whether to save global cellid information.
    
    """
    if os.path.exists(inputdata):
        ctype_df = pd.DataFrame()
        label_to_index = {}
        color_key = {}
        with open(inputdata, 'r') as f:
            for i, line in enumerate(f):
                h5ad_path = line[:-1]
                # get coordinate x,y and annotation
                adata = ad.read(h5ad_path)
                x = adata.obsm[position][:, 0]
                y = adata.obsm[position][:, 1]
                z = adata.obsm[position][:, 2]
                anno = adata.obs[label]

                # save anno dicts
                if i == 0:
                    label_to_index = {label: index for index, label in enumerate(adata.uns[colordict].keys())}
                    color_key = adata.uns[colordict]

                single_df = pd.DataFrame(
                    {'x': x, 'y': y, 'z': z, 'anno': anno})
                if withid:
                    slice_id = np.full(x.shape[0], int(i))
                    single_df['slice_id'] = slice_id
                    single_df['bin_id'] = single_df.index.astype('float')

                ctype_df = ctype_df.append(single_df)
                print("load ", h5ad_path, " done!")

            ctype_df['hex'] = ctype_df['anno'].map(color_key)
            ctype_df[['R', 'G', 'B']] = ctype_df['hex'].apply(
                lambda x: pd.Series([int(x[i:i+2], 16) for i in (1, 3, 5)]))
            ctype_df['label'] = ctype_df['anno'].map(label_to_index)
            ctype_df = convert_to_mm(ctype_df, resolution)

            if zdistance:
                ctype_df = add_zrandom(ctype_df, zdistance)
                print("add z ranom value done!")

            save_ply(outputdir, ctype_df, prefix,
                     "total", 0, withvalue, withid)
            print("save all celltype as ply done!")

        for label in label_to_index.keys():
            single_ctype_df = ctype_df[ctype_df['anno'] == label]
            save_ply(outputdir, single_ctype_df, prefix, str(
                label_to_index[label]), 0, withvalue, withid)
            print("save celltype: ", label, " as ply done!")


def make_geneexp_ply_data(inputdata, outputdir, prefix, position, genename, resolution, zdistance,   cutoff,  withvalue, withid, colorrange, cmap_name='rainbow4'):
    """
    The coordinates and gene expression levels are converted into point cloud files, and the gene expression levels are
    color mapped according to the specified color mapping table.

    参数：
    inputdata: String, a list text file storing all single-slice h5ad and corresponding z values
               after multiplexing and registration.
    outputdir: A string that specifies the output file path.
    prefix: A string specifying the output file prefix.
    position: String, used to read coordinates in h5ad.
    genename: String, used to read single gene expression data in h5ad.
    resolution: Integer array, saves the actual physical length of each axial coordinate unit 1 in the unified
                coordinate system of all slices, in nm.
    zdistance: Integer, saves the minimum value of the distance between all adjacent slices.
    cutoff: Floating point type, used to filter the expression threshold.
    withvalue: Boolean, indicating whether to save the expression intensity information.
    withid: Boolean, indicates whether to save global cellid information.
    cmap_name: String specifying the name of the colormap to use, defaults to 'rainbow4'.
    """

    if os.path.exists(inputdata):
        gene_df = pd.DataFrame()
        with open(inputdata, 'r') as f:
            for i, line in enumerate(f):
                h5ad_path=line[:-1]
                # get coordinate x,y and gene_expression

                adata = ad.read(h5ad_path)
                if issparse(adata.X):
                    if genename == "sum":
                        exp = adata.X.A.sum(axis=1)
                    else:
                        exp = adata[:, adata.var_names == genename].X.A
                        if exp.shape[1] == 0:
                            print("warning: ", genename, " not in ", h5ad_path)
                            continue
                else:
                    if genename == "sum":
                        exp = adata.X.sum(axis=1)
                    else:
                        exp = adata[:, adata.var_names == genename].X
                        if exp.shape[1] == 0:
                            print("warning: ", genename, " not in ", h5ad_path)
                            continue

                single_df = pd.DataFrame(adata.obsm[position])
                single_df.columns = ['x', 'y', 'z']
                single_df['exp'] = exp
                if withid:
                    slice_id = np.full(single_df.shape[0], int(i))
                    single_df['slice_id'] = slice_id
                    single_df['bin_id'] = single_df.index.astype('float')
                gene_df = gene_df.append(single_df)
                print("load ", h5ad_path, " done!")

            # filter null data
            gene_df = gene_df[gene_df['exp'] != 0]

            # filter low data
            gene_df = gene_df[gene_df['exp'] >= cutoff]
            gene_df=gene_df.reset_index()
            if gene_df.shape[0] == 0:
                print("error: ", genename, " not in all h5ad ")
            else:
                gene_df['cell_color'] = gene_df['exp']/gene_df['exp'].max()
                cmap = cc.cm[cmap_name]
                color_df = pd.DataFrame(value_to_color(
                    np.array(gene_df['cell_color']), colorrange,cmap), columns=['R', 'G', 'B'])
                gene_df = pd.concat([gene_df, color_df], axis=1)
                gene_df = convert_to_mm(
                    gene_df, resolution)

                if zdistance:
                    gene_df = add_zrandom(
                        gene_df, zdistance)
                save_ply(outputdir, gene_df, prefix, genename, 1, withvalue, withid)
                if genename == "sum":
                    print("save gene expression sum as ply done!")

                else:
                    print("save single gene: ", genename,
                          " expression as ply done!")
                    

def make_multi_geneexp_ply_data(inputdata, outputdir, prefix, position, genename_list, resolution, zdistance,  withvalue, withid, colorrange, color_list):
    """
    The coordinates and gene expression levels are converted into point cloud files, and the gene expression levels are
    color mapped according to the specified color mapping table.

    parameter：
    inputdata: String, a list text file storing all single-slice h5ad and corresponding z values after multiplexing
               and registration.
    outputdir: A string that specifies the output file path.
    prefix: A string specifying the output file prefix.
    position: String, used to read coordinates in h5ad.
    genename_list: List, used to read single gene expression data in h5ad.
    resolution: Integer array, saves the actual physical length of each axial coordinate unit 1 in the unified coordinate
                system of all slices, in nm.
    zdistance: Integer, saves the minimum value of the distance between all adjacent slices.
    cutoff: Floating point type, used to filter the expression threshold.
    withvalue: Boolean, indicating whether to save the expression intensity information.
    withid: Boolean, indicates whether to save global cellid information.
    cmap_name: String specifying the name of the colormap to use, defaults to []. 'rainbow4'.
    """

    if os.path.exists(inputdata):
        for i in range(len(genename_list)):
            gene_df = pd.DataFrame()
        with open(inputdata, 'r') as f:
            for i, line in enumerate(f):
                h5ad_path=line[:-1]
                # get coordinate x,y and gene_expression
                adata = ad.read(h5ad_path)

                if issparse(adata.X):
                    exp = adata[:, genename_list].X.A
                else:
                    exp = adata[:, genename_list].X

                # merge

                single_df = pd.DataFrame(adata.obsm[position])
                single_df.columns=['x','y','z']
                single_df[genename_list]=exp
                if withid:
                    slice_id = np.full(single_df.shape[0], int(i))
                    single_df['slice_id'] = slice_id
                    single_df['bin_id'] = single_df.index.astype('float')
                gene_df = gene_df.append(single_df)
                print("load ", h5ad_path, " done!")

            # filter null data
            filtered_gene_df = gene_df[gene_df[genename_list].any(axis=1) != 0]  
            filtered_gene_df['Classification'] = filtered_gene_df[genename_list].apply(classify_rows, axis=1) 
            if zdistance:
                filtered_gene_df = add_zrandom( filtered_gene_df, zdistance)
                filtered_gene_df = convert_to_mm(filtered_gene_df, resolution)
            
            color_dict = dict(zip(genename_list, color_list))
            cl=filtered_gene_df['Classification'].unique()

            # save to ply file
            for i in range(cl.shape[0]):
                sub_gene_df = filtered_gene_df[filtered_gene_df['Classification'] == cl[i]][['x', 'y', 'z']]
                if '#' in cl[i]:
                    multi_list = cl[i].split('#')
                    sub_gene_df['exp'] = \
                        filtered_gene_df[filtered_gene_df['Classification'] == cl[i]][multi_list].apply(lambda x: x.sum(), axis=1)
                    merge_color = color_dict[multi_list[0]]
                    for j in range(1, len(multi_list)):
                        merge_color = color_mix(merge_color, color_dict[multi_list[j]])
                    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ['#ffffff', merge_color])
                else:
                    sub_gene_df['exp'] = filtered_gene_df[filtered_gene_df['Classification'] == cl[i]][cl[i]]
                    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ['#ffffff', color_dict[cl[i]]])
                    
                sub_gene_df['cell_color'] = sub_gene_df['exp']/sub_gene_df['exp'].max()
                sub_gene_df = sub_gene_df.reset_index()
                color_df = pd.DataFrame(value_to_color(
                        np.array(sub_gene_df['cell_color']), colorrange, cmap), columns=['R', 'G', 'B'])
                sub_gene_df = pd.concat([sub_gene_df, color_df], axis=1)

                save_ply(outputdir, sub_gene_df, prefix, cl[i], 1, withvalue, withid)
                print("save single gene: ", i, cl[i], " expression as ply done!")
                
                print(cl[i], sub_gene_df.shape)


def save_ply(dir, ctype_df, prefix, label, ptype, withvalue=True, withid=True):
    """
    Save the coordinates and RGB to the point cloud ply file.

    parameter：
        dir: A string that specifies the output file path.
        ctype_df: pandas Data frame, saves coordinate and RGB information.
        prefix: A string specifying the output file prefix.
        label: A string that specifies the output file type.
        ptype:int, Used to specify the input file type.
        withvalue: bool, Used to specify whether to save label information.

    """
    pc_array = np.array(ctype_df[['x', 'y', 'z', 'R', 'G', 'B']])

    types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    if withvalue:
        if ptype == 1:
            pc_array = np.concatenate(
                (pc_array, np.array(ctype_df['exp']).reshape(-1, 1)), axis=1)
            types.append(('value', 'f4'))
        elif ptype == 0:
            pc_array = np.concatenate(
                (pc_array, np.array(ctype_df['label']).reshape(-1, 1)), axis=1)
            types.append(('label', 'u1'))
    if withid:
        pc_array = np.concatenate(
            (pc_array, np.array(ctype_df[['slice_id', 'bin_id']])), axis=1)
        types.append(('slice_id', 'u1'))
        types.append(('bin_id', 'f4'))
    points = []
    for i in range(pc_array.shape[0]):
        tmppt = []
        for j in range(pc_array.shape[1]):
            tmppt.append(pc_array[i, j])
        points.append(tuple(tmppt))
    vertex = np.array(points, dtype=types)
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    outputfile = dir+'/'+prefix+'_'+label+'.ply'
    PlyData([el], text=True).write(outputfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_in", help="input h5ad list with z value", type=str)
    parser.add_argument(
        "output_path", help="where to save ply files", type=str)
    parser.add_argument("type", help="data type to ply file",
                        type=str, choices=["celltype", "gene_exp","multi_gene_exp"])
    parser.add_argument(
        "-g", "--gene", help="gene name to ply file(additionally,sum for the sum of expression values for all genes)", type=str)
    parser.add_argument("-p", "--prefix", help="prefix for ply files", type=str,
                        default='test')
    parser.add_argument("-l", "--label", help="label in h5ad file for celltype data ,default='ctype_user'", type=str,
                        default='ctype_user')
    parser.add_argument("--colordict", help="label in h5ad file for celltype colormap dict data ,default='color_key'", type=str,
                        default='color_key')
    parser.add_argument("--position", help="label in h5ad file for bin position ,default='spatial'", type=str,
                        default='spatial')
    parser.add_argument("--colormap",
                        help="colormap name for expression value ,support colormaps from package colorcet,default='rainbow4'",
                        type=str, default='rainbow4')
    parser.add_argument("--colorrange", help="percent range of expression value for colormap,default='0,100'", type=str,
                        default='0,100')
    parser.add_argument("--cutoff", help=" gene expression threshold, default=0", type=float,default=0)
    parser.add_argument("-r", "--randomz", help="add random value to z,default=False",
                        action='store_true', default=False)
    parser.add_argument("-z", "--zdistance", help="minium distance between two neighbour slices, nm,default=20000",
                        type=int, default=20000)
    parser.add_argument("--resolution", help=" X,Y,Z values in nanometers,default='500,500,500'", type=str,
                        default='500,500,500')
    parser.add_argument("--gene_list", help=" gene name list at multi_gene mode", type=str)
    parser.add_argument("--color_list", help=" hex list for gene colormap at multi_gene_exp mode", type=str)
    parser.add_argument("-v", "--value", help="save celltype_label/gene_exp_value,default=False",
                        action='store_true', default=False)
    parser.add_argument("--id", help="save global ids, default=False ",
                        action='store_true', default=False)

    args = parser.parse_args()

    res = list(eval(args.resolution))
    if args.randomz:
        zdistance = args.zdistance
    else:
        zdistance = 0

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.type == "celltype":
        make_celltype_ply_data(args.data_in, args.output_path,
                               args.prefix,  args.position, args.label, args.colordict, res, zdistance, args.value, args.id)
    elif args.type == "gene_exp":
        crange=list(eval(args.colorrange))
        make_geneexp_ply_data(args.data_in, args.output_path,
                              args.prefix, args.position, args.gene,  res, zdistance, args.cutoff, args.value, args.id,  crange, args.colormap)
    elif args.type == "multi_gene_exp":
        crange=list(eval(args.colorrange))
        gene_list= args.gene_list.split(',')
        color_list = args.color_list.split(',')
        make_multi_geneexp_ply_data(args.data_in, args.output_path,
                              args.prefix, args.position, gene_list,  res, zdistance, args.value, args.id,  crange, color_list)


if __name__ == '__main__':
    main()
