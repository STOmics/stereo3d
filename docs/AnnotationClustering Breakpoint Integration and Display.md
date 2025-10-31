# **Annotation/Clustering Breakpoint Integration and Display**

### **Use Case: Annotation/Clustering Breakpoint Integration**

Use Case: Users have annotated/clustered h5ad data (h5ad has unified cell types or cluster labels, scanpy format) that needs to be integrated into the stereo3d analysis pipeline.

Requirement: The input h5ad files are already sorted in the correct order.

Tool: anno_adata_support.py; By default, it does not change the input h5ad colors.

##### Input Parameters:

`--matrix_path` (str, required)
File path: Cluster annotations are provided as H5AD files (AnnData objects) with spatial coordinates stored under the `obsm["spatial"]` key

`--output_path` (str, required)
Output Path

`--align_json_path` (str)
Alignment parameters file: In base pipline, it can be found in `02.register\01.align_mask\align_info.json` if it not provide, coordinates remain unmodified.

`--cut_json_path` (str)
Tissue mask parameters file: In base pipline, it can be found in `02.register\00.crop_mask\mask_cut_info.json` if it not provide, coordinates remain unmodified.

```
--cluster_key` (str)
Cluster label: default = "leiden"
The cluster label name , save in anndata file `.obs
```

##### Output file:

`11.tans_adata` :
Output H5AD file containing modified x/y coordinates and newly generated z-coordinates.

`12.adata_organ` :
The new 3D organ files.

For detailed information, please refer to the usage instructions:
https://github.com/STOmics/stereo3d/blob/dev/stereo3d/tools/README.md



##### Operation Steps:

1. Open the terminal, pull the code, set up the environment, and install packages

```
# Clone the development branch of the stereo3d repository
git clone https://github.com/STOmics/stereo3d.git

# Create a new conda environment named 'stereo3d' with Python 3.8
conda create --name=stereo3d python=3.8

# Activate the newly created conda environment
conda activate stereo3d

# Navigate into the cloned repository directory
cd stereo3d

# Install all required dependencies from the requirements file
pip install -r requirements.txt
```

2. Locate the tool tools/anno_adata_support.py and run it

If the input h5ad files do not have unified coordinates, use this example:

```
cd stereo3d/tools
python anno_adata_support.py -m D:\RD_Data\3D\test\ann_file -o D:\RD_Data\3D\output_aj -aj D:\RD_Data\3D\demo_output_20250416\02.register\01.align_mask\align_info.json -cj D:\RD_Data\3D\demo_output_20250416\02.register\00.crop_mask\mask_cut_info.json
```

If the input h5ad files already have unified coordinates, use this example:

```
cd stereo3d/tools
python anno_adata_support.py -m D:\RD_Data\3D\demo_output_20250416\06.color -o D:\RD_Data\3D\output_aj
```

3. Running interface and output results

Interface display after running completion

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/741aaf5e-01b5-4f1a-883b-972c2923d0fd.png)

Output results interface display

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/a0f04c93-5943-4525-8382-134fe5ed8594.png)

### **Use Case: Annotation/Clustering Effect Display**

Prerequisite: The h5ad files have unified coordinates and unified cell types or cluster labels

Operation Steps:

1. Open the terminal and navigate to the path where stereo3d_viewer.ipynb is stored

For example: If K3D_show.ipynb is stored in D:\code\3d, open the terminal, switch to the D drive, then cd into D:\code\3d

```
D:
cd  D:\code\3d
```

2. Install the Python 3.9 environment (supports jupyter), then install packages, and run jupyter notebook
 ```
   # Create a new conda environment named 'jupyter_env' with Python 3.9
   conda create -n jupyter_env python=3.9
   
   # Activate the conda environment
   conda activate jupyter_env
   
   # Install required packages: Jupyter Notebook, K3D visualization, NumPy, AnnData and seaborn
   pip install jupyter notebook k3d numpy anndata seaborn
   
   # Start Jupyter Notebook server
   jupyter notebook
 ```

3. A browser window will pop up, double-click stereo3d_viewer.ipynb

![截屏2025-07-18 17.26.32.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/f224191f-9e4b-4b15-8c0f-910eb5deafcf.png)

4. Fill in the path to the h5ad file

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/eea9c41a-2c3e-49eb-9d02-67d4572c8240.png)

5. Click "**Run this Cell and Advance**" to run the code. Click once to run one cell. The red boxes in the image below from left to right are: **Run this Cell and Advance**, **Interrupt the Kernel**, **Restart the Kernel**, **Restart the Kernel and Run All Cells**![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/2a597706-53c0-4288-882a-49f09d9ff640.png)
6. Display effect of the fruit fly demo data

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/a9b5c715-0781-4bf0-9d65-1054dfd556db.png" alt="image.png" style="zoom:50%;" /><img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/0c2649d0-867d-4f5c-838d-7014504c4b7f.png" alt="image.png" style="zoom:50%;" />



7. Press Ctrl+C in the terminal to exit jupyter notebook
8. If you need to adjust the point size, modify "point_size: float = 0.01" (the larger the value, the larger the points; location as shown below)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/043c4087-fc34-4a4a-a128-4d9036316e3f.png)

9. If you need to adjust the slice interval, modify "z_intervel = 0.02" (the larger the value, the larger the slice interval; location as shown below)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/J9LnW6jwaBM7VlvD/img/557195eb-f9cd-469d-829b-dd3ba095082c.png)