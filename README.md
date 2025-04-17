
<h1 align="center">
  <img src="docs/lamprey.gif" width=50% height=50%><br/>
</h1>

<h4 align="center">
  Stereo3D: Auto get stereo 3D data with Python
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
* ```--record_sheet``` Record sheet file. We provide you with a [sample](docs/E-ST20220923002_slice_records_20221110.xlsx), click for [detail](docs/extra.md).
* ```--tissue_mask``` The [saw](https://github.com/STOmics/SAW) output files(04.tissuecut), [details](docs/extra.md)
* ```--matrix_path``` The [saw](https://github.com/STOmics/SAW) input files(gene matrix)
* ```--output``` Output dir

Run ```python stereo3d --help``` for detail.

```shell
python stereo3d_with_matrix.py \
--matrix_path E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\01.gem \
--tissue_mask E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.mask \
--record_sheet E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\E-ST20220923002_slice_records_E14_16.xlsx \
--output E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\output
```
After running, the result you get is composed of multiple files and directories. For more detailed instructions, you need to visit [here](docs/extra.md)
