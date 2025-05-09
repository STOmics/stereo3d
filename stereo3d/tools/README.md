SUPPLYMENT TOOLS INTRODUCTION

### anno_adata_support tools

User-annotated results are incorporated into the stereo3D 
pipeline, where coordinate transformation is performed 
using parameters from "02.merge" to generate 
both transformed *.h5ad files and 3D organ outputs.


### Usage:
```bash
python anno_adata_support.py 
-m
D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\ann_file
-o
D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output
-aj
D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\02.register\01.align_mask\align_info.json
-cj
D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\02.register\00.crop_mask\mask_cut_info.json
```  


##### Input Parameters:  


- **`--matrix_path`** (str, required)  
  File path:
  Cluster annotations are provided as H5AD files (AnnData objects)
  with spatial coordinates stored under the `obsm["spatial"]` key
- **`--output_path`** (str, required)  
  Output Path
- **`--align_json_path`** (str)  
  Alignment parameters file:
  In base pipline, it can be found in `02.register\01.align_mask\align_info.json`
  if it not provide, coordinates remain unmodified.
- **`--cut_json_path`** (str)  
  Tissue mask parameters file:
  In base pipline, it can be found in `02.register\00.crop_mask\mask_cut_info.json`
  if it not provide, coordinates remain unmodified.
- **`--cluster_key`** (str)   
  Cluster label: default = "leiden"  
  The cluster label name , save in anndata file `.obs`


##### Output file:

- **`11.tans_adata`** :   
  Output H5AD file containing modified x/y coordinates and newly generated z-coordinates.
- **`12.adata_organ`** :  
  The new 3D organ files. 
    