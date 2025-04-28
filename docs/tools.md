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


## 2. Multi Tissue
```shell
python multi_tissue.py \
--mask_path D:\Desktop\stereo3d_1tom\SS200000122BL_B1\SS200000122BL_B1-2.tif \
--matrix_path D:\Desktop\stereo3d_1tom\SS200000122BL_B1\SS200000122BL_B1.gem.gz \
-output D:\Desktop\stereo3d_1tom\output\SS200000122BL_B1
```

| Name        | Description                 | Importance | Dtype  |
|-------------|-----------------------------|------------|--------|
| matrix      | The path of matrix file     | Required   | string |
| mask        | The path of tissue cut file | Required   | string |
| output      | The output path             | Required   | string |