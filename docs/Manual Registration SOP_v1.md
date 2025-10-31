# **Manual Registration SOP_v1**

## 1 **Basic Introduction**

● Application Scenario: When the Stereo3D automated registration results do not meet requirements, perform manual registration operations on the automatically registered files, then reconnect to the Stereo3D pipeline to output new results.

● Definition of "Reference Image" and "Registration Image": The reference image is the image used as the coordinate reference and does not undergo transformation itself; the registration image is the image that will be transformed. By default, according to the order in the slice record sheet, the preceding image is the reference image, and the following image is the registration image.

● Note: When viewing with Fiji software and importing into TrakEM2, the images are displayed sorted by filename (according to the default alphanumeric order). Therefore, it is recommended to name the files according to the order in the slice record sheet (e.g., image1.tif, image2.tif) to ensure the correct preceding and subsequent relationship.

## 2 **Tool Description and Preparation**

Since the image and the matrix have a one-to-one correspondence, and the matrix transformation reuses the relationship from image registration, it is sufficient to check whether the image registration is correct.

The output results in the 02.register folder include 00.crop_mask (cropped tissue mask) and 01.align_mask (registered tissue mask).

| Tool | Version      | Introduction                                                 | Download Link                              | Purpose                                                      | BGI Pan Download Link                                        |
| :--- | :----------- | :----------------------------------------------------------- | :----------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Fiji | ImageJ 1.54f | Fiji is a free, open-source image processing software based on ImageJ, supporting fluorescence microscopy, 3D reconstruction, cell counting, etc. | https://imagej.net/software/fiji/downloads | ● View registration results ● Manually modify registration results | https://bgipan.genomics.cn/#/link/OvDFhNolvDYrxCO8l8V0 Access Code: RdpK |

## 3 **Operation Steps**

### 3.1 **Create Project**

● Create TrakEM2: Open Fiji software and create an empty TrakEM2;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/c1d1aa22-4925-4356-851a-44ed8a6f18fe.png" alt="image" style="zoom:80%;" />

● Select Directory: In the "Select storage folder" window, select the 01.align_mask folder (the directory containing the images);

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/455c6505-b78b-45b7-be99-e2149ef09827.png" alt="image" style="zoom: 80%;" />

● Remove Other Files: Move the align_info.json file out of the 01.align_mask folder (**The folder imported into TrakEM2 should only contain images, no other files**);

● Import Folder: Drag the 01.align_mask folder into the "Canvas" (the largest image display area); Click "OK" in the "Import directory" window; In the "Slice separation" window, check the "Lock stack" option;



<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/3b9a3824-473c-4efa-ab47-086582b7776b.png" alt="image.png" style="zoom: 33%;" /> 

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/0e9a2c8c-86e2-4d68-af92-8c2c3e3d9c8f.png) 

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/5169b0e1-1dc0-4bde-bf98-1737249ea6bd.png) 

### 3.2 **Registration Preparation**

● Adjust Image Position: If you need to adjust the display position of the image in the "Canvas", first place the mouse on the "Preview thumbnail", hold down "Ctrl", and scroll the mouse wheel to adjust the image display position in the "Canvas"; You can also click and drag the red box in the "Preview thumbnail" to move it up, down, left, or right until the image display is suitable;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/ecb287a8-945a-4780-9373-6fad0962de57.png" alt="image.png" style="zoom: 33%;" />

● Unlink Layers: In the "Patches" window, unlink the layers (so that the reference image can subsequently undergo positional transformation);



![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/68889406-2646-462f-a27c-180d84ddb7df.png) 

● Switch to the "Layers" Window: In the "Layers" window, the number of indicator bars on the left equals the number of files in the imported folder. Scrolling the mouse wheel will cause the indicator bars to turn green alternately, and the "Canvas" will display different images in turn;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/08b8781c-a459-4194-87cc-9df6b87f508a.png?x-oss-process=image/crop,x_0,y_0,w_1217,h_817/ignore-error,1" alt="image.png" style="zoom: 50%;" />

● Image Selection: The indicator bar for the reference image is red, and the indicator bar for the image to be adjusted is green; Select the indicator bar of the image to be adjusted (it turns green when selected); Press the "Shift" key and click on another indicator bar in the left panel, then the indicator bar for the reference image will turn red;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/0e3867d4-4945-42ab-9f89-b79f43df30d2.png" alt="image.png" style="zoom:50%;" /> 

● Check Registration Results: Scroll the mouse wheel to check the registration status between consecutive images.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/jP2lRYKDKaYJ6O8g/img/d43c124d-259b-40b8-ae72-db16f97403a8.png" alt="image.png" style="zoom: 50%;" />

### 3.3 **Rigid Registration**

Image pairwise rigid registration can be completed through operations such as flipping, translation, and rotation. After registration is completed, output the results according to section 3.5.

First, click on the green image (the registration image). A white rectangular box will appear around it. Then right-click and select Transform → Transform(affine) to start manual rigid registration.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/4f99acc6-c6d3-426f-b95e-da842999700d.png" alt="image.png" style="zoom:50%;" />

● Flip Operation

If you need to flip the green image (i.e., the image to be adjusted), right-click on the green image, select "Specify transform...", change the value of "Scale in Y" to -1, and click "OK".

Note: Scaling operations are also performed by changing the values of "Scale in X" and "Scale in Y" under "Specify transform...". However, in principle, the unit pixels of the two input masks should correspond to the same physical scale. If the scales of the registration images are found to be different, the input image sizes should be checked.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/ba88f2a4-75d1-4a6d-be9b-144db7337e6f.png)

● Translation and Rotation

Place the mouse on the green image, left-click and drag the green image to achieve translation; Rotate the handle of the clock pointer to achieve rotation around the center of the pointer. You can perform coarse adjustment first, then fine-tune. After registration is completed, right-click and click "Apply Transform".

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/b4236f63-4029-4917-95c1-ce37f2b40e37.png)

### 3.4 **Elastic Registration**

If the images have local deformations (such as bending or stretching of biological tissues) and rigid registration cannot meet the requirements, elastic registration can be used to register the deformed areas. In elastic registration, anchor points can "constrain" the transformation of local areas: the area between adjacent anchor points will be smoothly adjusted based on the correspondence of the anchor points, ensuring alignment accuracy under complex deformations. Note that elastic registration should be performed on the basis of completed rigid registration.

If you have just completed the rigid registration of the corresponding images in TrakEM2 and have not closed the workspace, you can skip (1) and (2) below and directly proceed with (3) and subsequent operations in the same interface.

(1) Create Project

The process is the same as in section 3.1 Create Project.

(2) Registration Preparation

The process is the same as in section 3.2 Registration Preparation.

(3) Elastic Registration Operation

The user will complete the registration of the deformed parts by selecting anchor points and moving them.

○ First, click on the green image (the registration image). A white rectangular box will appear around it. Then right-click and select Transform → Transform(non-linear).

○ Anchor the Image: Hold down "Shift" and use the left mouse button to click on the operation points (referred to as "anchor points") on the image for registration. If not necessary, the distance between points should not be too close (for example, in the case of registering mouse embryo cluster images to Blockface images, generally no more than 15 points). It is recommended to anchor the image outline first, then anchor the center, to avoid excessive changes during adjustment (as shown below).

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/b7127229-2c00-4c36-a487-300cef1b3f51.png)

○ Select Anchor Points for Local Adjustment: Click the left mouse button and drag the anchor points so that the same tissue areas of the green image and the red image overlap on the canvas, achieving elastic registration, as shown below.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/ef17feee-3de7-43f4-8308-90ac43308b2d.png) 

○ Other Operations: If you want to restart the registration operation during adjustment, right-click and select "Cancel Transform"; If you want to delete a single anchor point, press "Ctrl+Shift" and click on the anchor point.

○ After registration is completed, right-click and click "Apply Transform";

### 3.5 **Output Registration Results**

After manual registration is completed, save the registration relationships and the registered images. (Number of adjusted images = Number of registration relationships)

● Save Registration Relationships

Right-click, select "Project", click "Save as...", and name the file sn.xml (sn = filename of the adjusted image)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/186dd66d-e930-4938-b943-b3b64f6aa74b.png)

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/5f4afaf8-7ec0-4457-8a29-08be00a2ba6c.png" alt="image.png" style="zoom: 80%;" />

● Save Registered Images

Right-click, select "Export", click "Make flat image…";

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/67087922-7aea-41d2-a015-bfd97bce586c.png)

In the pop-up "Choose" window, change the "Type" option to "RGB Color" or "8-bit grayscale" (depending on the nature of the imported image); Change the "Export" option to "Save to file" (as shown below). Click "OK", and save to the same location as above;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/40323c7e-5aea-4a74-8558-84df35a8bbb8.png" alt="image.png" style="zoom: 67%;" />

After the file is automatically saved, find the file in the save path and rename it to "sn_manual".

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/cfd70e34-585d-4585-b63a-5180b3676c37.png" alt="image.png" style="zoom:80%;" />

## 4 **Reconnect to Stereo3D Pipeline**

After manually registering the images, you need to reconnect to the Stereo3D pipeline to generate new results.

● The file structure after registration is as follows: The trakem2 folder generated during manual registration should be deleted; align_info.json is in the 01.align_mask folder;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/QvjnA3jo2QZ2bOXo/img/ce275b02-3950-4aad-8fea-042fc4676d33.png" alt="image.png" style="zoom:80%;" />

● In the result file's 02.register folder, create a new subfolder named 02.manual, and place the registration relationship file sn.xml inside it;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/bc1ee4fa-e612-4e69-ad90-4de0d05ffc8f.png" alt="image.png" style="zoom:80%;" />

● Place the exported registered image sn_manual.tif into the 01.align_mask folder, and remove the original sn.tif;

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/a40c7e5c-a0b3-4975-a39b-76b592188175.png" alt="image.png" style="zoom:80%;" />

● Delete the result folders from 03.gem to 07.organ

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2M9qP5jz0WawgO01/img/98a6a519-db7a-49e2-aaad-5d90db86e9ad.png" alt="image.png" style="zoom:80%;" />

● Use the `--overwriter` parameter. The command is as shown below, which generates new 3D results based on the manual registration results.

text

```
python stereo3d_with_matrix.py \
--matrix_path D:\RD_Data\3D\Drosophila_melanogaster_demo\00.gem \
--tissue_mask D:\RD_Data\3D\Drosophila_melanogaster_demo\00.mask \
--record_sheet D:\RD_Data\3D\E-ST20220923002_slice_records_E14_16.xlsx \
--output D:\RD_Data\3D\demo_output_20250416 \
--overwriter 0
```