# **3Dslicer Manual Modification of mesh/organ SOP_v1**

## 1.**Basic Introduction**

For unsatisfactory mesh or organ results from the Stereo3D automated pipeline, 3D Slicer can be used for manual modification. (Since the model results do not affect other file results, there is no need to reconnect to the Stereo3D pipeline)

● Introduction to commonly used tool windows in 3D Slicer software

The five windows under "Welcome to Slicer", with the specific functions of each window shown in Figure 1.

Data: Data window;

Models: Model transparency adjustment;

Segment Editor: Volume manual tool adjustment;

Segmentations: Volume transparency adjustment window;

Surface Models: Model Maker is the window for model reconstruction.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/r4mlQ5bxX8M9Zlxo/img/9a88f0e7-c674-474b-9046-ac5086484d77.png" alt="image.png" style="zoom: 67%;" />

Figure 1 Example of commonly used windows

● In the 3D view of 3D Slicer, click the "center the 3D" button in the upper left corner to reset the model position. Scrolling the mouse wheel zooms in/out the model (holding the left mouse button allows viewing the model from any angle; holding the right mouse button or mouse wheel adjusts the model size).

## 2.**Tool Description and Preparation**

| Tool      | Version        | Introduction                                                 | Download Link                | Purpose                 |
| :-------- | :------------- | :----------------------------------------------------------- | :--------------------------- | :---------------------- |
| 3D Slicer | v4.11.20210226 | Open-source medical image analysis software supporting 3D visualization, segmentation, registration of multimodal data, widely used in research and clinical settings. | https://download.slicer.org/ | ● Modify model contours |

## 3.**Operation Steps**

### 3.1 **Import Data**

Open 3D Slicer, directly drag the mesh/organ file to be modified (format .obj) into 3D Slicer as a model;

Note: The save path for the obj file cannot contain Chinese characters

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/6f63a705-cf5f-4528-b6a5-fe0aefad6cc8.png)

Figure 2 Loading obj data into 3D Slicer software

Adjust the view: In the 3D view, click the "center the 3D" button in the upper left corner to reset the model position, then scroll the mouse wheel backward to zoom in on the model (hold the left mouse button to move the model freely; hold the right mouse button or mouse wheel to adjust the model size.)

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/e70b3108-8bf9-4c05-ac25-aec6da041ac0.png" alt="image.png" style="zoom:80%;" />

Figure 3 Adjusting the model view position within 3D Slicer

### 3.2 **Convert models mode to editable segment mode**

Switch to the Data window, select the model to be modified, right-click and select "Convert model to segmentation node", as shown in Figure 4; Select the converted segmentation, right-click and select "export visible segment to binary labelmap", as shown in Figure 5. This step converts the model to volume and then to an editable mode.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/r4mlQ5bxX8M9Zlxo/img/ad4f31c7-6730-4ccb-bd41-4176985bb344.png)

Figure 4 Convert model to segmentation

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/r4mlQ5bxX8M9Zlxo/img/4e53faf2-b579-4315-9424-4251d40016e2.png)

Figure 5 Example of converting volume to binary labelmap

### 3.3 **Transparency Adjustment (Optional, does not affect overall operation)**

The methods for adjusting transparency for Model and Segment are as follows:

1. Model Transparency Adjustment

Left-click to select the file to adjust transparency, then click the Models window to switch to the models interior as shown in Figure 6, then left-click to select the models file. Wait for the Visibility mode to change from gray to adjustable black mode, check the Visibility window, and drag the slider as shown in Figure 7. The effect can be viewed in the 3D view.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/ea07c88f-c2be-4081-8248-d57b1a04e5b3.png" alt="image.png" style="zoom:80%;" />

Figure 6 Switching model transparency

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/353e52a2-a9db-40b8-8a1e-962d2d702445.png)

Figure 7 Example after model transparency adjustment

2. Segment Transparency Adjustment

The logic for segment transparency adjustment is the same as for model transparency adjustment. Select the "segmentations" window, and adjust the slider in the Display module;

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/4df77f0a-2ee4-4c62-8b9f-039fb793c1ef.png)

Figure 8 Segment transparency adjustment

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/0b2ea87c-03ed-4c89-a5cb-cf29cdbcbcb5.png)

Figure 9 Example after segment transparency adjustment

### 3.4 **Manual Filling of Internal Cavities**

Switch to the Segment Editor window. At this time, the manual adjustment tools in the Effects interface including Paint, Draw, etc., are grayed out, as shown in Figure 10;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/r4mlQ5bxX8M9Zlxo/img/e6bc773d-669a-4dee-b176-98dd855f048b.png" alt="image.png" style="zoom: 80%;" /> 

Figure 10 Example of segment editor window

After selecting the correct Segmentation and Master volume, the manual tools become colored and adjustable, as shown in Figure 11;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/299af816-07c0-4e00-9d70-42da1c9fe7a5.png" alt="image.png" style="zoom: 25%;" />

Figure 11 Introduction to the segment editor interface

At this point, you can adjust the 3 views in the 3D view to check the cavities in 2D (Figure 12). The cavities are the parts that need adjustment;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/02243474-4bf5-428b-a856-682e420ec05d.png" alt="image.png" style="zoom: 50%;" />

Figure 12 Example of scrolling

During adjustment, use the Paint, Draw, and Erase tools frequently, with Paint being the most commonly used. For fine adjustments in the early stage, you can select Paint and choose the Sphere brush option. The Diameter should be selected based on the largest cavity in the 3D view. You can enable the Show 3D button to view the coverage area of the brush diameter (as shown in Figure 14). When using the brush, you need to turn off the 3D view, otherwise it may lag or crash directly. After using the spherical fill, you can uncheck Sphere brush and fill on a single slice. The final result is shown in Figure 15.

If you find many "burr-like", unsmooth areas, you can use the Erase tool to remove them. Remember to turn off the 3D view when erasing.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/r4mlQ5bxX8M9Zlxo/img/8ad71ee7-bc42-4552-8367-0726b46966c1.png" alt="image.png" style="zoom: 80%;" />

Figure 13 Example of adjustment tools

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/3739d751-b1cc-4df4-8771-b65224992e83.png" alt="image.png" style="zoom: 25%;" />

Figure 14 Example of brush radius view

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/9168481e-af31-4108-9597-f6af8f1d64b3.png" alt="image.png" style="zoom: 33%;" />

Figure 15 Example after the brain region is filled cleanly

### 3.5 **Convert segment to model**

Switch to the Surface Models and Model Maker windows in sequence (Figure 16), set the parameters for forming the model from the volume in order, and finally click apply (Figure 17). When downsampling, it needs to be set according to the file size. The final result should ideally not exceed 30M after TAR compression.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/8303d1db-4574-485b-b3bf-9932abde221f.png" alt="image.png" style="zoom: 50%;" />

Figure 16 Example of switching to the model generation interface

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/50ea4809-07d3-4782-829d-14f09fd1da19.png" alt="image.png" style="zoom: 50%;" />

Figure 17 Example of model formation parameter settings

### 3.6 **Save as obj file**

Press the shortcut key ctrl+S, select the model file generated in the previous step, and set the save format to .obj. Set the save path. If the operation is not completed in one session, you can select all files and click the "��" button in the upper left corner. This option will output a .mrb format file and store all files in the project, similar to a Photoshop psd file.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOkW800a43O4BX/img/55085ecb-00c1-405c-b7a3-2f4df3067011.png" alt="image.png" style="zoom:67%;" />

Figure 18 Example of save format output