# Multiple Automatic Splitting Solution for One Chip

In spatial transcriptomics experiments, when multiple tissue sections from the same embedding block are placed on a spatial chip or multiple tissue sections are attached to a single chip, the Stereo3D intelligent assistant tool `multi_tissue.py` can be used to split the multi-tissue samples on the chip into independent single tissues for subsequent import into the Stereo3D analysis pipeline for 3D reconstruction.

### 一、Tool Path: tools/multi_tissue.py

The purpose is to automatically split one-chip-multiple-tissue data. After splitting, the preceding and subsequent relationships need to be manually modified in the record sheet.

Process: Download the code, configure the environment, enter the tools directory under the stereo3d directory, run multi_tissue.py (to split the one-chip-multiple-tissue data), manually modify the record sheet, return to the stereo3d directory, and run stereo3d_with_matrix.py to obtain 3D results.

**Notes**:

1. During program execution, avoid any operations on the output path. At the end of the program, intermediate results will be adjusted for storage path and naming to adapt to stereo3D. Intermediate operations may affect the normal operation of the program.
2. The naming in the ssDNA_ChipNo column of the record sheet must be consistent with the mask file naming.
3. In Windows systems, tmp temporary files need to be manually deleted by the user. In Linux systems, tmp temporary files are deleted by the program by default.
4. In Windows systems, after running, there may be this output: "'gzip' is not recognized as an internal or external command, operable program or batch file." This does not affect the results.

### 二、multi_tissue.py Running Example

```
git clone https://github.com/STOmics/stereo3d.git # Clone the repository named stereo3d
conda create --name=stereo3d python=3.8 # Create a conda virtual environment named stereo3d with Python version 3.8
source activate stereo3d # Activate the conda virtual environment named stereo3d
cd stereo3d # Enter the stereo3d directory
pip install -r requirements.txt # Install project dependencies according to the requirements.txt file
cd stereo3d/tools # Enter the tools subdirectory under the stereo3d directory
python multi_tissue.py --mask_path D:\Desktop\stereo3d_1tom\FP200000449TL_C3.tif --matrix_path D:\Desktop\stereo3d_1tom\FP200000449TL_C3.gem.gz --output D:\Desktop\stereo3d_1tom\output\FP200000449TL_C3  # Run the multi-tissue analysis script, --mask_path specifies the mask path, --matrix_path specifies the matrix data path, -output specifies the output path
```



Input Introduction:

| Parameter | Description            | Type | Required |
| :-------- | :--------------------- | :--- | :------- |
| matrix    | Input matrix path      | str  | Yes      |
| mask      | Input tissue mask path | str  | Yes      |
| output    | Output path            | str  | Yes      |

Output Results Introduction:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557L101zwno8/img/6ed62727-af10-466b-b024-983635802cb8.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557L101zwno8/img/a65511fe-6d6e-42da-adb3-08e2e1ded10d.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557L101zwno8/img/50ee4879-dc24-43c1-bfe7-dcd4892b6bb3.png)

ID Representation File Example:

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/pLdn557L101zwno8/img/a21099aa-b490-4641-8da4-acf8e4e1e4eb.png)

### 三、Reconnecting to stereo3d

After running multi_tissue.py and automatically splitting the data, it needs to be reconnected to the stereo3d pipeline to obtain stereo3d results.

1. Manually modify the record sheet; the naming must be consistent with the mask file naming.

   Example as follows:

   Mask file naming:

   ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenvep0P10Zzloy/img/38d03cd7-14da-424a-8a90-8f2a515c2869.png)

   Slice record sheet: modify ssDNA_ChipNo to be consistent with the mask file naming; modify the preceding and subsequent order.

   ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenvep0P10Zzloy/img/a80f3254-062b-4254-b9b9-0ed9c0e48056.png)

2. Run stereo3d

After running multi_tissue.py, the current directory is the tools subdirectory under the stereo3d directory. Use `cd ..` to return to the stereo3d directory (because multi_tissue.py and stereo3d_with_matrix.py are not in the same directory level, it is necessary to return to the directory where stereo3d_with_matrix.py is located). The running example is as follows:

```
cd .. 
python stereo3d_with_matrix.py --matrix_path D:\RD_Data\3D\test_sawv8\output\test_out\gem --tissue_mask D:\RD_Data\3D\test_sawv8\output\test_out\mask --record_sheet D:\RD_Data\3D\mouse_brain.xlsx --output D:\RD_Data\3D\test_data
```