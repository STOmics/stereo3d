## 1. SAW_HUB

```shell
python saw_hub.py \
-input D:\data\stereo3d\input \
-record D:\code\stereo3d\docs\E-ST20220923002_slice_records_20221110.xlsx \
-stain ssDNA \
-output D:\data\stereo3d\output \
-saw_version 7
```

|  Name   | Description                  | Importance | Dtype  |
|  ----  |------------------------------|------------|--------|
| input  | SAW output dir               | Required   | string |
| record  | Input record sheet path      | Required   | string |
| output  | The output path              | Required   | string |
| saw_version  | The version of SAW (7 or 8)  | Required   | int    |
| stain  | The stain tech (ssDNA or HE) | Required   | string |

## 3. spateo-3d
Here, we use the spateo framework to build a simple process to input multiple adjacent slices of h5ad files and output the aligned h5ad files and 3D model files. The output format is consistent with the input of spateo-viewer, so the results can be rendered and viewed on it.
* Test data1, [Drosophila embryos](https://db.cngb.org/stomics/flysta3d/download/)
* Test data2, [Mouse Embryo](https://db.cngb.org/stomics/mosta/download/)

### setup
* [spateo](https://github.com/aristoteleo/spateo-release)
    ```shell
    # python=3.9
    pip install pymeshfix
    pip install pyacvd
    pip install scanpy
    pip install spateo-release==1.1.1
    ```
* [spateo-viewer](https://github.com/aristoteleo/spateo-viewer)
    ```shell
    git clone https://github.com/aristoteleo/spateo-viewer.git
    cd spateo-viewer
    pip install -r requirements.txt
    ```
   _Note:_ vtk==9.2.2 can solve ```TypeError: Could not find a suitable VTK type for <U54```

### usage
* spateo
    ```shell
    python spateo_3d.py \
    -input D:\data\stereo3d\spateo\gy\E14-16h_a_count_normal_stereoseq.h5ad \
    -output D:\data\stereo3d\spateo\gy\E14-16h \
    -z_step 1 \
    -slice slice_ID \
    -cluster annotation \
    -cluster_pts 1000
    ```
  
  |  Name   | Description                                                                                                                                                                                                                                                                        | Importance | Dtype  |
  |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------|
  | input  | File or directory path. <br>File path: all slices are in one file; <br>Directory path: all slices are in one directory (the file name must contain numbers to reflect their relationship)                                                                                          | Required   | string |
  | output  | Result save path                                                                                                                                                                                                                                                                   | Required   | int    |
  | z_step  | The distance between slices can be adjusted according to the rendering effect in the spateo-viewer                                                                                                                                                                                 | Optional   | string |
  | cluster  | The field name where the clustering information is located in the h5ad file                                                                                                                                                                                                        | Optional   | int    |
  | slice  | Name the field in the h5ad file that represents the relationship between slices before and after                                                                                                                                                                                   | Optional   | string |
  | cluster_pts  | When there are too many points to describe the 3D tissue expression information, <br>downsampling can be used, and the sampled data is used for process analysis. <br>Here is the number of points sampled for each category,<br>Please use with caution when you have few points. | Optional   | int    |

* spateo-viewer

  * open
    ```shell
    python stv_explorer.py --port 1234
    ```
  * show, [more](https://github.com/aristoteleo/spateo-viewer/blob/main/usage/spateo-viewer.pdf)