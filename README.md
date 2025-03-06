
<h1 align="center">
  <img src="docs/lamprey.gif" width=25% height=25%><br/>Stereo3D
</h1>

<h4 align="center">
  Auto get stereo 3D data with Python
</h4>

## Description
The technologies for obtaining square-bin and [cell bin](https://github.com/STOmics/cellbin2/tree/main ) data of a single slice are relatively mature. Considering the organization's 3D structure, obtaining spatio-temporal 3D data can be solved by investing a lot of manpower, but the cost is high, the cycle is long, and it is very unfriendly to users. We are solving this problem by combining deep learning and image processing technology.

## Installation
There are options:

- Platform agnostic installation: [Anaconda](#anaconda)
- Platform specific installation: [Ubuntu](#ubuntu), [macOS](#macos), [Windows](#windows)
### Anaconda

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```bash
# python3.8
conda create --name=stereo3d python=3.8
source activate stereo3d

# pip on python3.8
pip install -r requirements.txt
```
## Usage
Before run the pipeline, You need to confirm the following four parameters:
* ```--record``` Record sheet file. We provide you with a [sample](docs/E-ST20220923002_slice_records_20221110.xlsx), click for [detail](docs/extra.md).
* ```--saw List``` Input [saw](https://github.com/STOmics/SAW) file, [details](docs/extra.md)
* ```--block``` Input block face files, [details](docs/extra.md)
* ```--output``` Output dir

Run ```python stereo3d --help``` for detail.

```
python stereo3d.py \
-r /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liuhuanlin/01.code/stereo3d/docs/E-ST20220923002_slice_records_20221110.xlsx \
-b /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/luqin3/Blockface_20221110_mouse_embroyo_rename \
-s /jdfssz2/ST_BIGDATA/Stomics/warehouse/prd/ods/STOmics/ShenZhen_projectData/Analysis_Result/E-ST20220923002 \
/jdfssz2/ST_BIGDATA/Stomics/warehouse/prd/ods/stomics/analysisResult/E-ST20220923002 \
-o /hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/01.cellbin/liuhuanlin/02.data/temp/bf_s3d
```
After running, the result you get is composed of multiple files and directories. For more detailed instructions, you need to visit [here](docs/extra.md)
