# 1. **Significance of 3D Reconstruction**

Three-dimensional (3D) reconstruction is one of the research hotspots in the current life sciences field. By constructing 3D atlases, it is possible to restore the overall distribution of cells and molecules from the massive transcriptomic data of cells. This provides key assistance in solving many unresolved mysteries in research, such as parsing organ functions, exploring the mechanisms of embryonic development, studying the pathogenesis of cancer and other complex diseases, and tracing species evolutionary history.

This technology breaks the limitations of traditional two-dimensional observation, allowing researchers to understand the relationships between gene expression, cell interactions, and biological phenotypes in a dimension closer to the native spatial architecture of biological entities. It opens up new perspectives and pathways for both basic life science research and clinical translation (e.g., discovery of disease diagnostic biomarkers, design of targeted therapies).

# 2. **STOmics 3D Reconstruction Solution Full Process and Feature Highlights**

Traditional spatial transcriptomics technologies only provide two-dimensional information from tissue sections, unable to fully present the three-dimensional structure of biological samples and the spatial relationships between cells. Even with high-resolution gene expression localization, dedicated 3D reconstruction tools are still needed to address the following challenges:

- Accurate spatial registration of consecutive tissue sections.
- Integration and normalization of multi-slice data.
- Three-dimensional visualization of cell types and gene expression patterns.
- Reconstruction of cell trajectories and developmental processes across sections.

Stereo3D is a 3D reconstruction analysis platform developed by BGI-STOmics. It can integrate multi-slice spatial information to construct 3D gene expression atlases at the tissue and organ level, providing new perspectives for developmental biology, disease mechanism research, and tissue engineering.

The core advantages of Stereo3D make it a powerful research tool:

- **High Versatility:** Flexibly adapts to various scenarios including image-only, matrix-only, or combined image and matrix data, meeting the analysis needs of different data types.
- **High Efficiency and Low Cost:** Significantly reduces labor costs and shortens the analysis cycle (The Drosophila Demo can produce results in less than ten minutes on a computer with 32GB RAM).

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/d8a4d81a-5e5c-4ebc-902b-0b88532cf79e.png)

Figure 1 Full Process of the 3D Reconstruction Solution

## 2.1 **Wet Lab Section Introduction (For Reference Only, Link Provided, No Detailed Description, Linked to Official Website)**

In the overall workflow of Stereo3D, the wet lab section is the key step for obtaining raw data. This experimental part is based on BGI's self-developed Stereo-seq technology, capturing mRNA information from tissue sections via the spatial-temporal chip. The experimental process includes precise sectioning of fresh tissue, mounting onto the spatial-temporal chip, followed by reverse transcription and cDNA amplification, recording gene expression data for each spot on the chip. BGI's supporting kit contains standardized reagents, ensuring experimental stability and data reliability.

To learn more about wet lab-related products and technical details, please visit the BGI official website: https://www.bgi.com/.

## 2.2 **Data Analysis Process Introduction**

The Stereo3D data analysis process is centered on automation and standardization, efficiently transforming raw data into 3D reconstruction results. In terms of input, it supports GEM/GEF format gene expression matrices, tissue mask images, and a slice record sheet documenting slice positions.

After the process starts, it first preprocesses the tissue mask images to achieve spatial alignment and construct a unified 3D coordinate system. It then performs basic analysis such as matrix transformation and annotation. Finally, it outputs registered tissue mask images, H5AD data, 3D models (OBJ), etc., helping researchers quickly gain 3D spatial insights and solving the problems of low efficiency and high labor costs associated with traditional methods.

## 2.3 **Automated Solutions**

The 3D reconstruction solution currently offers two automated methods: Stereo3D and Spateo.

The standardized Stereo3D process achieves 3D reconstruction by integrating histological images and gene expression matrices, while also supporting 3D reconstruction based solely on images or solely on matrices. If the sample morphology is intact and needs to be combined with tissue morphology analysis, Stereo3D is preferred.

The difference between Spateo and Stereo3D is that it does not rely on histological images. It is entirely based on gene expression matrices, using statistical generative models to achieve image-independent 3D reconstruction. If sample morphology varies significantly, Spateo is recommended.

### 2.3.1 **Stereo3D**

#### 2.3.1.1 **Installation and Usage**

Stereo3D is an open-source Python tool that supports deployment on Linux and macOS systems. The following are the quick start steps:

1. Prepare the installation environment

Please install Anaconda software first ([Anaconda Official Documentation](https://docs.anaconda.com/anaconda/install/)), then perform the following operations:

2. Install Stereo3D

```
# Clone the repository and navigate to the directory
git clone https://github.com/STOmics/stereo3d
cd stereo3d

# Create and activate the Conda environment
conda create --name=stereo3d python=3.8
conda activate stereo3d

# Install dependencies
pip install -r requirements.txt
```



3. Run stereo3d_with_matrix.py

```
python stereo3d_with_matrix.py \
--matrix_path E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.gem \
--tissue_mask E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.mask \
--record_sheet E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\E-ST20220923002_slice_records_E14_16.xlsx \
--output E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\output
```

For more installation details, parameter configuration, and tutorial cases, please visit the [Stereo3D GitHub repository](https://github.com/STOmics/stereo3d). If you encounter any issues, please submit feedback or participate in discussions via GitHub Issues.

#### 2.3.1.2 **Input Parameter Introduction**

If you have the standard output results from SAW, you need to first use the small tool `saw_hub.py` in Stereo3D (see details in 3.2.1 Stereo3D Input Standardization) to convert the data into the standard input format for Stereo3D.

Table 1 Stereo3D Standard Input Parameter Introduction

| **Input**    | **Description**                                              | **Required** | **Data Type** | **Remarks**         |
| :----------- | :----------------------------------------------------------- | :----------- | :------------ | :------------------ |
| matrix_path  | Standard format gene expression matrix, supports raw.gef/gef/gem.gz | Required     | string        | /                   |
| tissue_mask  | Tissue mask image, tif format                                | Required     | string        | /                   |
| record_sheet | Obtained from the experimental side, records slice positions, correspondence between preceding and subsequent slices | Required     | string        | /                   |
| output       | Result save path                                             | Required     | string        | /                   |
| registration | The process performs registration by default. If the input data is already registered and no additional algorithmic registration is needed, use parameter `--registration 0` | Optional     | int           | /                   |
| overwriter   | If the automated registration result of Stereo3D does not meet requirements, perform manual registration operations on the automatically registered files, then feed back into the Stereo3D process to output new results, use parameter `--overwriter 0` | Optional     | int           | Example see 2.4.1.1 |
| align        | If only the matrix is input, matrix reconstruction results can be generated, outputting only the registered H5AD and organ mesh, use parameter `--align paste` | Optional     | string        | Example see 3.2.3.2 |

#### 2.3.1.3 **Standard Output File Introduction**

| **Output File** | **Description**                                              |
| :-------------- | :----------------------------------------------------------- |
| 02.register     | Registered tissue mask images after alignment                |
| 03.gem          | Spatial expression matrix after registration                 |
| 04.mesh         | 3D mesh model reconstructed from clustered point clouds      |
| 05.transform    | Annotated H5AD file containing spatial coordinates and cell metadata |
| 06.color        | H5AD file with unified color mapping for visualization       |
| 07.organ        | Segmented organ-specific mesh models                         |

#### 2.3.1.4 **Viewing Results**

After data standardization and running `stereo3d_with_matrix.py` to output results, the results can be viewed in the following ways.

##### 2.3.1.4.1 **Viewing Registration Results**

Since the images and matrices have a one-to-one correspondence, and the matrix transformation reuses the image registration relationship, it is sufficient to check whether the image registration is correct.

The output result in the `02.register` folder contains `00.crop_mask` (cropped tissue masks) and `01.align_mask` (registered tissue masks).

| Tool | Version      | Description                                                  | Download Link                              | Purpose                                                      | BGI Pan Download Link                                        |
| :--- | :----------- | :----------------------------------------------------------- | :----------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Fiji | ImageJ 1.54f | Fiji is a free, open-source image processing software based on ImageJ, supporting fluorescence microscopy images, 3D reconstruction, cell counting, etc. | https://imagej.net/software/fiji/downloads | ● View registration results ● Manually modify registration results | https://bgipan.genomics.cn/#/link/OvDFhNolvDYrxCO8l8V0 Extraction code: RdpK |

Note: When using Fiji software to view, importing into TrakEM2 displays them sorted by filename (according to the default alphanumeric order). Therefore, it is recommended to name the files according to the order in the slice record sheet (e.g., image1.tif, image2.tif) to ensure the correct sequence relationship.

● **Create Project:** Open Fiji software and create an empty TrakEM2.

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/c1d1aa22-4925-4356-851a-44ed8a6f18fe.png)

● **Select Directory:** In the "Select storage folder" window, choose the `01.align_mask` directory (the directory containing the images).

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/455c6505-b78b-45b7-be99-e2149ef09827.png" alt="image" style="zoom:80%;" />

● **Remove Other Files:** Move the `align_info.json` file from the `01.align_mask` folder out of this folder (The folder imported into TrakEM2 can only contain images, no other files).

● **Import Folder:** Drag the `01.align_mask` folder into the "Canvas" (the largest image display area). In the "Import directory" window, click "OK". In the "Slice separation" window, check "Lock stack".

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/3b9a3824-473c-4efa-ab47-086582b7776b.png" alt="image.png" width="65%">
</div> 

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/0e9a2c8c-86e2-4d68-af92-8c2c3e3d9c8f.png" width="50%">
</div> 

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/5169b0e1-1dc0-4bde-bf98-1737249ea6bd.png" width="50%">
</div> 



● **Adjust Image Position:** To adjust the display position of the image in the "Canvas", first place the mouse on the "Preview thumbnail", hold down "Ctrl", and scroll the mouse wheel to adjust the display position of the image in the "Canvas". You can also click the red box of the "Preview thumbnail" and move it up, down, left, or right until the image display is suitable.

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/ecb287a8-945a-4780-9373-6fad0962de57.png" width="60%">
</div> 

● **Unbind Layers:** In the "Patches" window, unbind the layers (so that the reference image can subsequently undergo positional transformations).

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/68889406-2646-462f-a27c-180d84ddb7df.png" width="50%">
</div> 

● **Switch to "Layers" Window:** In the "Layers" window, the number of indicator bars on the left equals the number of files in the imported folder. Scrolling the mouse wheel will cause the indicator bars to turn green alternately, and the "Canvas" will display different images in turn.

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/08b8781c-a459-4194-87cc-9df6b87f508a.png?x-oss-process=image/crop,x_0,y_0,w_1217,h_817/ignore-error,1" alt="image.png" width="60%">
</div> 

● **Image Selection:** The indicator bar for the reference image is red, and the indicator bar for the image to be adjusted is green. Select the indicator bar of the image to be adjusted (it turns green when selected). Press the "Shift" key and click on other indicator bars in the left column; the reference image's indicator bar will then turn red.

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/0e3867d4-4945-42ab-9f89-b79f43df30d2.png" width="60%">
</div> 

● **View Registration Results:** Scroll the mouse wheel to view the registration between consecutive slices.

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/d43c124d-259b-40b8-ae72-db16f97403a8.png" width="60%">
</div> 

##### 2.3.1.4.2 **H5AD Result Visualization**

The output result `06.color` is an H5AD file with unified coordinates and labels. Use the small tool `stereo3d_viewer.ipynb` in Stereo3D for visualization. For specific operations, please refer to [《Annotation/Clustering Breakpoint Integration and Display》](https://github.com/STOmics/stereo3d/blob/dev/docs/AnnotationClustering%20Breakpoint%20Integration%20and%20Display.md) **--- Use Case: Annotation/Clustering Effect Display**

##### 2.3.1.4.3 **Mesh/Organ Visualization**

The output result `04.mesh` is the entire tissue model, and `07.organ` is the organ model. Use 3D Slicer for visualization (models are stored in OBJ format).

| Tool      | Version        | Description                                                  | Download Link                | Purpose                                      |
| :-------- | :------------- | :----------------------------------------------------------- | :--------------------------- | :------------------------------------------- |
| 3D Slicer | v4.11.20210226 | Supports 3D reconstruction, segmentation, and visualization of multi-dimensional medical data like CT and MRI. | https://download.slicer.org/ | ● Model visualization ● Modify model outline |

Note: The OBJ file save path cannot contain Chinese characters.

● **Import File:** Drag the files from the `04.mesh` folder into 3D Slicer.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/657f711c-b4ee-44d7-af77-8279ed9d0f50.png?x-oss-process=image/crop,x_9,y_11,w_736,h_429/ignore-error,1)

● **View Results:** In the 3D view, click the "center the 3D" button in the upper left corner to reset the model position. Scrolling the mouse wheel zooms in/out the model. (Hold the left mouse button to view the model from any angle; hold the right mouse button or mouse wheel to adjust the model size.)

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/a67650b5-cc97-45ba-ba68-534b8a820aa6.png" width="60%">
</div> 

● **Mesh and Organ Visualization:** Drag the files from the `07.organ` folder into 3D Slicer to display them together with the mesh.

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/afd8a515-930a-4049-95d3-78a46f3fc827.png" width="60%">
</div> 

 <div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/47a72e4e-59c3-4c56-91b5-0e8a785ab271.png" width="60%">
</div> 



● **Adjust Model Display Color:** In the "Models" module, double-click the color window to bring up the "Slicer" window, select a color to change the display color.

 <div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/9a662521-0611-45ea-99be-4b15cd1d7e5a.png" width="60%">
</div>  

 <div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/bbcb91ab-f3de-4948-93c5-ec158e36b784.png?x-oss-process=image/crop,x_1544,y_51,w_1007,h_689/ignore-error,1" alt="image.png" width="60%">
</div>   

● **Show/Hide OBJ Parts:** Click the eye icon. When the eye is closed, the part is hidden; when open, it is displayed.

 <div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/b7658629-3f34-4df9-baef-9d10a37a811b.png" width="60%">
</div>   

### 2.3.2 **Spateo**

Spateo focuses on computational analysis for 3D reconstruction of spatial transcriptomics data and spatial pattern mining (e.g., slice registration, cell trajectory construction). Spateo-viewer is the accompanying visualization tool, specifically designed to display the analysis results from Spateo. It presents complex 3D spatial structures, gene expression patterns, etc., in an interactive manner. Together, they form a complete "analysis - visualization" workflow.

#### 2.3.2.1 **Installation and Usage**

Spateo is a tool developed based on Python. The following are the quick start steps:

1. Prepare the installation environment

Please install Anaconda software first ([Anaconda Official Documentation](https://docs.anaconda.com/anaconda/install/)), then perform the following operations:

2. Install Spateo and Spateo-viewer

   spateo

```
# Clone the repository and navigate to the directory
git clone https://github.com/STOmics/stereo3d
cd stereo3d/stereo3d/tools

# Create and activate the Conda environment
conda create --name=spateo python=3.9
conda activate spateo

# Install dependencies
pip install pymeshfix
pip install pyacvd
pip install scanpy
pip install spateo-release==1.1.1
```

   Spateo-viewer

```
# Clone the repository and navigate to the directory
git clone https://github.com/aristoteleo/spateo-viewer.git
cd spateo-viewer

# Install dependencies
pip install -r requirements.txt

#Note: vtk==9.2.2 can solve TypeError: Could not find a suitable VTK type for <U54
```



3. Run spateo_3d.py and stv_explorer.py

```
python spateo_3d.py \
-input D:\RD_Data\3D\spateo_data\E14-16h_a_count_normal_stereoseq.h5ad \
-output D:\RD_Data\3D\spateo_data\output \
-z_step 1 \
-slice slice_ID \
-cluster annotation \
-cluster_pts 1000
```



```
python stv_explorer.py --port 1234
```



● For more installation details, parameter configuration, and tutorial cases, please visit the [spateo-release GitHub](https://github.com/aristoteleo/spateo-release) repository and the [spateo-viewer GitHub](https://github.com/aristoteleo/spateo-viewer) repository.

● Spateo article: https://www.cell.com/cell/fulltext/S0092-8674(24)01159-0

#### 2.3.2.2 **Input Parameter Introduction**

| **Input**   | **Description**                                              | **Required** | **Data Type** |
| :---------- | :----------------------------------------------------------- | :----------- | :------------ |
| input       | File or directory path. File path: All slices are in one file; Directory path: All slices are in one directory (filenames must contain numbers to reflect inter-slice relationships). | Required     | string        |
| output      | Result save path                                             | Required     | string        |
| z_step      | Adjust the distance between slices based on the rendering effect in spateo-viewer | Optional     | int           |
| cluster     | The field name in the H5AD file where the clustering information is stored | Optional     | string        |
| slice       | The field name in the H5AD file that represents the sequential relationship between slices | Optional     | string        |
| cluster_pts | When the number of points used to describe 3D tissue expression information is too large, downsampling can be used, and the sampled data is used for process analysis. Here is the number of points sampled per class. Use with caution if the point count is low. | Optional     | int           |

#### 2.3.2.3 **Output File Introduction**

| **Output File** | **Description**                                              |
| :-------------- | :----------------------------------------------------------- |
| h5ad            | AnnData object processed by count normalization and spatial alignment, containing gene expression matrix and spatial coordinate information. |
| matrices        | Raw or preprocessed gene expression matrix files             |
| mesh_models     | Segmented organ-specific mesh models                         |
| pc_models       | Aligned organ-specific point cloud model containing coordinates and attributes of discrete cells/sites in 3D space |
| plot            | Visualized results file                                      |
| trans_h5ad      | AnnData after normalization, alignment, and secondary processing, with analysis data |

#### 2.3.2.4 **Viewing Results**

For details, please visit the [spateo-viewer GitHub](https://github.com/aristoteleo/spateo-viewer) repository.

## 2.4 **Manual Refinement Solutions**

If you are not satisfied with the results of the Stereo3D automated pipeline, you can manually modify the results and then feed them back into the pipeline.

#### 2.4.0.1 **Manual Registration**

If the registration result in the Stereo3D automated pipeline is incorrect, you can perform manual registration and then feed it back into Stereo3D to output new automated pipeline results. For specific operations, please refer to this document: [《Manual Registration SOP_v1》](https://github.com/STOmics/stereo3d/blob/dev/docs/Manual%20Registration%20SOP_v1.md)

#### 2.4.0.2 **Manual Mesh Modification**

If you are not satisfied with the mesh or organ results from the Stereo3D automated pipeline, you can use 3D Slicer for manual modification. (Since the model results do not affect other file results, there is no need to feed them back into the Stereo3D pipeline).

[《3D Slicer Manual Mesh/Organ Modification SOP_v1》](https://github.com/STOmics/stereo3d/blob/dev/docs/3Dslicer%20Manual%20Modification%20of%20meshorgan%20SOP_v1.md)

# 3 **Application Case Introduction**

## 3.1 **Test Data Introduction**

| Species Tissue | Download Path                                                | File Size |
| :------------- | :----------------------------------------------------------- | :-------- |
| Drosophila     | https://bgipan.genomics.cn/#/link/wSnaxQhJ6i5RTPcYvtgI Extraction password: FJrY | 90 MB     |

### 3.1.1 **Drosophila Automated Pipeline Demonstration**

a. Environment Configuration

```
# Clone the repository and navigate to the directory
git clone https://github.com/STOmics/stereo3d
cd stereo3d

# Create and activate the Conda environment
conda create --name=stereo3d python=3.8
conda activate stereo3d

# Install dependencies
pip install -r requirements.txt
```



b. Code Execution

```
python stereo3d_with_matrix.py \
--matrix_path E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.gem \
--tissue_mask E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.mask \
--record_sheet E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\E-ST20220923002_slice_records_E14_16.xlsx \
--output E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\output
```



c. Log Display

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/5bc75b9e-1d15-4c3a-a6de-a9db3f0ea65d.png" alt="image.png" style="zoom:80%;" />

d. Output Result Display

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/06219592-7a05-41de-a11f-fcaa2674e456.png" alt="image" width="20%" style="margin-right: 10px;">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/d2d4ab6a-d3af-4e51-8bb0-4902ec4364c3.png" alt="image" width="20%" style="margin-left: 10px;">
</div>

<div align="center">Pairwise Registration Result Display</div>

<div align="center">
  <img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/8f80c78c-f3f9-4e42-87c4-41c949739d9b.gif" alt="image3.gif" width="30%">
</div>

<div align="center">Model Result Display</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/STOmics/stereo3d/dev/docs/drosophila_melanogaster.gif" alt="drosophila_melanogaster.gif" width="70%">
</div>

<div align="center">Post-Clustering Effect Display</div>

## 3.2 **Scenario Introduction**

### 3.2.1 **Image and Matrix Scenario**

The image and matrix scenario is the standardized process for Stereo3D. For details, please refer to section 3.1.1 Drosophila Automated Pipeline Demonstration.

### 3.2.2 **Image-Only Scenario**

When only images are available, Stereo3D can be used to generate tissue reconstruction results. The `--matrix_path` input can be any path; an actual matrix file is not required as input.

**Usage**


```
python stereo3d_with_matrix.py \
--matrix_path E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.gem \
--tissue_mask E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.mask \
--record_sheet E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\E-ST20220923002_slice_records_E14_16.xlsx \
--output E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\output
```



**Output File**

| **File Name** | **Description**                                         |
| :------------ | :------------------------------------------------------ |
| 02.register   | Registered tissue mask images after alignment           |
| 03.gem        | Spatial expression matrix after registration            |
| 04.mesh       | 3D mesh model reconstructed from clustered point clouds |
| 05.transform  | empty                                                   |
| 06.color      | empty                                                   |
| 07.organ      | empty                                                   |

If a specific structure within the tissue needs to be displayed separately, that structure can be clustered into one category and processed through Stereo3D to generate the structure model. For more details, please submit feedback or participate in discussions via GitHub Issues.

### 3.2.3 **Matrix-Only Scenario**

When only matrix files are available without images, you can use Stereo3D for reconstruction. Alternatively, you can choose to use Spateo for reconstruction; see section 2.3.2 Spateo for details.

#### 3.2.3.1 **Stereo3D --- Tissue Reconstruction**

If you want to obtain the overall tissue mesh, tissue segmentation can be performed based on the matrix. The generated tissue segmentation mask can then be used as input for Stereo3D, ultimately yielding its standard output results.

For tissue segmentation using the matrix, you can utilize cellbin2. For specific operations, see the link: https://github.com/STOmics/cellbin2/blob/main/cellbin2/config/demos/only_matrix.json

After obtaining the tissue segmentation results from the matrix, you have the standard input for Stereo3D. The subsequent process can be referred to in section 3.1.1 Drosophila Automated Pipeline Demonstration.

#### 3.2.3.2 **Stereo3D --- Matrix Reconstruction**

If you only want to obtain the matrix reconstruction results, you can use the `paste` parameter in Stereo3D to output the registered H5AD and organ mesh.

**Usage**


```
python stereo3d_with_matrix.py \
--matrix_path D:\RD_Data\3D\Drosophila_melanogaster_demo\00.gem \
--record_sheet D:\RD_Data\3D\E-ST20220923002_slice_records_E14_16.xlsx \
--output D:\RD_Data\3D\Drosophila_melanogaster_demo\output \
--align paste
```



**Input Parameters**

| **Name**     | **Description**                                              | **Importance** | **Dtype** |
| :----------- | :----------------------------------------------------------- | :------------- | :-------- |
| matrix_path  | The path of matrix file                                      | Required       | string    |
| tissue_mask  | The path of tissue mask file                                 | Required       | string    |
| record_sheet | Record sheet file. We provide you with a [sample](https://file/D:/code/lym/3d/stereo_dev/stereo3d/docs/docs/E-ST20220923002_slice_records_20221110.xlsx), click for [detail](https://file/D:/code/lym/3d/stereo_dev/stereo3d/docs/docs/extra.md) | Required       | string    |
| output       | The output path                                              | Required       | string    |
| align        | Using the paste method to achieve alignment                  | Required       | string    |

**Output File**

| **File Name** | **Description**                                        |
| :------------ | :----------------------------------------------------- |
| 06.color      | H5AD file with unified color mapping for visualization |
| 07.organ      | Segmented organ-specific mesh models                   |

## 3.3 **Intelligent Assistant Tools Introduction**

Stereo3D is also equipped with a series of practical small tools to adapt to different scenarios. The following will specifically introduce the application of these tools in various scenarios.

### 3.3.1 **Stereo3D Input Standardization**

If you have the standard output results from SAW, in addition to manually standardizing the input, you can also use the small tool `saw_hub.py` in Stereo3D to convert the data into the standard input format for Stereo3D. `saw_hub.py` prioritizes obtaining `raw.gef` from the SAW standard output results. If not available, it obtains `gem.gz` and automatically converts `gem.gz` to `gef`.

**Usage**

```
python saw_hub.py \
-input D:\data\stereo3d\input \
-record D:\code\stereo3d\docs\E-ST20220923002_slice_records_20221110.xlsx \
-stain ssDNA \
-output D:\data\stereo3d\output \
-saw_version 7
```



**Input Parameters**

| **Name**    | **Description**                      | **Importance** | **Dtype** |
| :---------- | :----------------------------------- | :------------- | :-------- |
| input       | SAW output directory                 | Required       | string    |
| record      | Input record sheet path              | Required       | string    |
| output      | The output path                      | Required       | string    |
| saw_version | The version of SAW (7 or 8)          | Required       | int       |
| stain       | The staining technique (ssDNA or HE) | Required       | string    |

**Output File**

| **File Name** | **Description**                                              |
| :------------ | :----------------------------------------------------------- |
| gem           | The [SAW](https://github.com/STOmics/SAW) input files (gene matrix) |
| mask          | The [SAW](https://github.com/STOmics/SAW) output files (<SN>_<staintype>_tissue_cut.tif), [details](https://stereotoolss-organization.gitbook.io/saw-user-manual-v8.2/analysis/outputs/count-outputs) |

### 3.3.2 **Multiple Tissues on One Chip Scenario Processing**

Due to small tissue size, multiple sections are mounted on one spatial-temporal chip, or multiple tissues are embedded in one block and then sectioned and mounted, resulting in multiple tissues on one chip. For such data, the experimental side needs xxxxxx (to be added later); on the data processing side, it needs to be split into individual tissues before being input into the Stereo3D pipeline for reconstruction. For data processing, there are manual splitting solutions and automated splitting solutions.

a. Manual Splitting Solution primarily uses the tools StereoMap and SAW. For specific operations, please refer to [《Multiple Manual Splitting Solution for One Chip》](https://github.com/STOmics/stereo3d/blob/dev/docs/Multiple%20Manual%20Splitting%20Solution%20for%20One%20Chip.md).

b. Automated Splitting Solution uses the tool `multi_tissue.py` within Stereo3D. For specific operations, please refer to [《Multiple Automatic Splitting Solution for One Chip》](https://github.com/STOmics/stereo3d/blob/dev/docs/Multiple%20Automatic%20Splitting%20Solution%20for%20One%20Chip.md).

### 3.3.3 **Annotation/Clustering Result Breakpoint Integration Function**

`anno_adata_support.py` is a tool specifically designed for spatial transcriptomics data analysis, used to integrate already annotated/clustered H5AD format data (Scanpy standard) as a breakpoint into the Stereo3D analysis pipeline. This tool seamlessly integrates external annotation data based on the original Stereo3D analysis results, generating new Stereo3D results while retaining all intermediate results from the original analysis pipeline.

For specific operations, please refer to [《Annotation/Clustering Breakpoint Integration and Display》](https://github.com/STOmics/stereo3d/blob/dev/docs/AnnotationClustering%20Breakpoint%20Integration%20and%20Display.md) --- **Use Case: Annotation/Clustering Breakpoint Integration**

### 3.3.4 **Adding Images Without Matrix for Reconstruction**

Adding images without matrix information to improve the 3D model is feasible within the Stereo3D pipeline. Completely fill in the slice record sheet with the SNs used for reconstruction, ensuring the correct sequence relationships between consecutive slices, and finally run `stereo3d_with_matrix.py`.

```
python stereo3d_with_matrix.py \
--matrix_path E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\01.gem \
--tissue_mask E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\00.mask \
--record_sheet E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\E-ST20220923002_slice_records_E14_16.xlsx \
--output E:\3D_demo\Drosophila_melanogaster\00.raw_data_matrix\Drosophila_melanogaster_demo\output
```



# 4 **Q&A**

1. If the number of samples is too small, or the sample quality is poor, should they be discarded?
   - **Answer:** If the number of samples is too small, it's best to ensure the optimal samples are used for the formal experiment, guaranteeing key positions are correct (preferably, the client confirms the start and stop points for chip mounting, as they know their requirements best). Poor sample quality doesn't significantly impact reconstruction, but it greatly affects subsequent data analysis, e.g., diffusion, low capture rate, insufficient gene count. It's recommended not to use poor-quality samples.
2. In 3D reconstruction, there is a non-mandatory requirement: the number of sequenced genes in adjacent slices (bin50) should not differ by more than 30%. If different bin sizes are used, should corresponding adjustments be made?
   - **Answer:** Yes, different bin sizes require corresponding adjustments.
3. Which is more recommended for 3D reconstruction, Stereo3D or Spateo?
   - **Answer:** Spateo's input is an annotated matrix, while Stereo3D uses image (ssDNA) registration. They are completely different strategies and haven't been specifically evaluated against each other; it depends on user choice. Intuitively, one could use image registration first and then use downstream tools for fine-tuning.
4. What does the registration score evaluate? What score is considered good, and how to judge if the registration result is correct?
   - **Answer:** The registration score evaluates the registration result between two adjacent slices. Data from mouse embryo projects score around 84%, which is quite good. Some Drosophila data have registration scores of 60%, but the registration results are correct. Actually, if the gap between two slices is large and the morphological difference is significant, the registration score might be low. If the score is low, it's best to visually inspect the registration result and make a judgment. (This requires the client to understand the tissue morphology. It was also suggested earlier to use morphological atlases, MRI, or CT to determine tissue morphology; provide the state of the sample before and after embedding, specifically ensuring the embedding direction has no significant deviation from the intended reconstruction orientation).
5. Is it feasible to add a few ssDNA images without matrix data into the 3D reconstruction automated pipeline?
   - **Answer:** Yes, adding ssDNA images to improve the 3D model is feasible within the automated pipeline.
6. (Duplicate of Q1) If the number of samples is too small, or the sample quality is poor, should they be discarded?
   - **Answer:** (Same as A1) If the number of samples is too small, it's best to ensure the optimal samples are used for the formal experiment, guaranteeing key positions are correct (preferably, the client confirms the start and stop points for chip mounting, as they know their requirements best). Poor sample quality doesn't significantly impact reconstruction, but it greatly affects subsequent data analysis, e.g., diffusion, low capture rate, insufficient gene count. It's recommended not to use poor-quality samples.
7. From the perspective of high-impact journals, tumor size dimensions, and the aesthetic appeal of the modeled results, how many slices are appropriate?
   - **Answer:** The number of slices is related to tissue thickness. It also depends on the target tissue/organ, the scientific question, the hotspots of the species/tissue being studied, and the inherent size of the tissue. For example, if the tissue is very small (can be fully sectioned in 4 slices), ensure the entire tissue structure is captured; there's no need to cut more. From a journal perspective, focus on the research hotspots. Also, cutting more slices increases the overall project scale and investment, naturally yielding more output. But with limited funding, focus on whether the sectioned area can cover the region of interest, and subsequent integration or mapping can be done; fewer slices can be cut and integrated with previously published 3D structures.
8. Comparison of application scenarios, advantages, and disadvantages between low resolution (few slices) and high resolution (many slices)?
   - **Answer:** For tumor data, if heterogeneity is the focus, and if tumor growth structure is the concern, high resolution might not be necessary. During the earlier publication period (2023), single-cell level wasn't achievable, so low-resolution transcriptome structures were chosen. Now, high resolution can achieve single-cell level, describing a different granularity. If only bin100 is done, smaller structures cannot be identified. Stereo-seq now can achieve single-cell level, giving researchers more angles to consider; they can choose. More slices aren't always better; it's more important to consider from the researcher's perspective whether this is useful for solving their problem. If funding allows, consider combining other technologies to make the entire project design more comprehensive and rigorous, not limited to the number of slices. Ensure the slices meet the technical requirements for publishing better papers.
9. For a 3D project, which staining would be recommended, ssDNA or HE?
   - **Answer:** If it meets the requirements for cellbin, ssDNA is more stable.
10. If there is a batch of consecutive sections, but the section thickness is not the same (e.g., some are 10 microns, some are 5 microns), how should the Z-interval value be set in the Meta information of the slice record sheet?
    - **Answer:** The `Z_index` column in the slice record sheet needs to be adjusted. This column represents the cumulative Z-axis distance for each slice from the start, not the interval from the previous slice. For example, if the first slice is 5 microns, enter 5; if the second slice is 10 microns, enter 15 microns. (The mesh will be constructed based on the z-interval values).

# 5 **Appendix**

● If you are interested in Stereo3D and want to learn more, please check the related articles:

3D Mouse Embryo Article:
Gastrulation Article: https://www.sciencedirect.com/science/article/abs/pii/S1934590925001420?via%3Dihub

Spateo Article: https://www.cell.com/cell/fulltext/S0092-8674(24)01159-0

Spatial Transcriptomics Technology Stereo-seq 3D Reconstruction Solution, BGI STOmics Official Website Video Introduction: https://www.stomics.tech/resources/Videos/1893.html

Spatial Transcriptomics Technology Stereo-seq Facilitates 3D Atlas Construction, BGI STOmics Official Website Video Introduction: https://www.stomics.tech/resources/Videos/1892.html

● List of Third-Party Software Used by Stereo3D

| Software Name | Download Link                                                | Version        | Purpose                                                      |
| :------------ | :----------------------------------------------------------- | :------------- | :----------------------------------------------------------- |
| ImageStudio   | https://www.stomics.tech/service/stomics-software-download.html | 3.*            | ● Quality control for imaging data, serving SAW registration |
| SAW           | https://www.stomics.tech/service/stomics-software-download.html | V8, V7 & prior | ● Completes registration between transcriptome and imaging data ● Performs tissue segmentation based on imaging data |
| Stereo3D      | https://github.com/STOmics/stereo3d                          | v0.0.1         | ● Registers multiple adjacent slices ● Unifies coordinate system, generates 3D model - tissue envelope ● Outputs registered matrices across multiple slices |
| 3D Slicer     | https://download.slicer.org/                                 | v4.11.20210226 | ● Modify model outline                                       |
| Cloud Compare | https://www.danielgm.net/cc/                                 | v2.12.4        | ● Modify internal point cloud of cell annotations            |
| Blender       | https://www.blender.org/download/                            | v4.0           | ● Result visualization, rendering                            |
| Excel         | -                                                            | -              | ● Make a cutter flow meter                                   |