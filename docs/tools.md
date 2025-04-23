## SAW_HUB

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
