# **Multiple Manual Splitting Solution for One Chip**

### 一. **Background Introduction**

Due to the small size of tissues, multiple tissue sections or multiple tissues from one embedding block are placed on a spatial chip, resulting in multiple tissues on one chip. For such data, it needs to be split into individual tissues before being input into the stereo3d pipeline for reconstruction.

The overall process involves using stereomap to perform manual lasso selection on the matrix gef for individual tissues, outputting Geojson, which is then fed into the saw reanalyze lasso pipeline to obtain multiple individual tissue matrix files (gem.gz) and corresponding mask.tif files.

### 二. **Tools**

| Name          | Introduction                                                 | Download Link                                                |
| :------------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **StereoMap** | StereoMap is a desktop software that provides essential interactive analysis functions for exploring your Stereo-seq data. | https://www.stomics.tech/service/stomics-software-download.html |
| **SAW**       | Stereo-seq Analysis Workflow (SAW) is a command-line data analysis pipeline tool for processing Stereo-seq spatial data. | https://www.stomics.tech/service/stomics-software-download.html |

### **Precautions**

1. The input for stereomap can be gef or gef+rpi. gef+rpi is recommended as it allows lasso selection based on the image, resulting in more accurate mask results.
2. The naming format and directory structure of the manually split results do not comply with the stereo3d input format and need to be manually modified by the user.
3. The record sheet also needs corresponding modifications; the preceding and subsequent relationships must be manually adjusted by the user.
4. The naming in the ssDNA_ChipNo column of the record sheet must be consistent with the mask file naming.

### 三. **StereoMap Operation Steps**

Use stereomap to perform manual lasso selection on the matrix gef for individual tissues and output Geojson.

1. Data Acquisition: Obtain gef and rpi from the saw output.
2. Import Data: Open stereomap, click Visual Explore, select gef+rpi, click Confirm to complete the import.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/a5d0228a-f42c-403f-849b-55c8fdc51584.png?x-oss-process=image/crop,x_0,y_0,w_1741,h_1093/ignore-error,1" alt="image.png" style="zoom: 33%;" />

Figure 1 Import gef+rpi

3. Adjust Layers: First, lower the transparency of the matrix layer to see the image.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/eabdbae0-329b-4188-baa3-2e2686b67eaf.png" alt="image.png" style="zoom: 50%;" />

Figure 2 Adjust Layer Transparency

4. Select Tissues: Click the lasso tool "<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/428eafa3-5679-411a-92c8-a2465bb3cd65.png" alt="image.png" style="zoom: 80%;" />", hold down the Ctrl key, aim the "+" on the page at the tissue to be selected, hold down the left mouse button to start selecting, and release the left mouse button to stop. To undo, you can only click "<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/719656dc-f920-494e-b100-b42dda801fbb.png" alt="image.png" style="zoom:80%;" />" to start over.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/e10de60d-e045-496d-be03-eae9c70b5978.png" alt="image.png" style="zoom: 33%;" />

Figure 3 Selecting Tissue

5. Save Single Tissue Annotation Result: After selecting a tissue, press the "Enter" key, a pop-up window will appear, click Save, customize the Label name and Group name for the tissue classification, and finally click Submit. As shown below:

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/7eb8f7d2-eb43-4a08-9568-9dee2951c2c1.png" alt="image.png" style="zoom:80%;" />

Figure 4 Pop-up window, click Save

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/9fe61c4f-aa7f-49e6-bcb9-686032d4b6ec.png" alt="image.png" style="zoom:80%;" />

Figure 5 Custom Labels

6. Save GeoJSON: Repeat steps 4-5 above until all tissues are selected. In the Groups interface, click "...", select "GeoJSON to lasso targeted area", and export all annotation results of the entire groups.

<img src="https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/YdgOk27oRyJQgq4B/img/247deff3-be6a-4be0-b5c6-5d234bdf78ba.png" alt="image.png"  />

Figure 6 Save GeoJSON

### 四. **SAW Reanalyze Lasso Pipeline**

Feed the GeoJSON into the saw reanalyze lasso pipeline to obtain multiple individual tissue matrix files (gem.gz) and corresponding mask.tif files. For detailed introduction, please refer to (saw user manual -- data reanalysis -- [Matrix Lasso Module](https://www.stomics.tech/service/saw_8_1/docs/shi-yong-jiao-cheng/secondary-analysis.html#矩阵套索))

If using bin GEF for lasso, run the analysis as follows:

The --bin-size parameter can accept a list to generate expression matrix files for multiple bin sizes at once.

```
saw reanalyze lasso \
  --gef=/path/to/input/GEF \
  --lasso-geojson=/path/to/lasso/GeoJSON \
  --bin-size=1,20,50 \
  --output=/path/to/output/lasso
```



The lasso output based on bin GEF is as follows:

```
lasso
├── <label1>
│    ├── SN.<label1>.label.gef  ##lasso GEF of bin1
│    └── segmentation
│        ├── SN.lasso.<bin_size_list[0]>.<label1>.gem.gz  ##GEM of lasso area of different bin sizes
│        ...
│        ├── SN.lasso.<bin_size_list[n]>.<label1>.gem.gz
│        └── SN.lasso.<label1>.mask.tif  ##mask image of lasso area
└── <label2>
    ├── ...
    └── ...
```



If using cellbin GEF for lasso, run the analysis as follows:

```
saw reanalyze lasso \
  --cellbin-gef=/path/to/input/cellbin/GEF \
  --lasso-geojson=/path/to/lasso/GeoJSON \
  --output=/path/to/output/lasso
```



The lasso output based on cellbin GEF is as follows:

```
lasso
├── <label1>
│    └── SN.<label1>.label.cellbin.gef  ##cellbin GEF of lasso area
└── <label2>
    └── ...
```



A running example is shown below:

```
/storeData/USER/data/07.STOmics_FAS/chenjinghao/tools/saw-8.1.3/bin/saw reanalyze lasso --gef /storeData/USER/data/01.CellBin/00.user/liyumei/tmp/register/SS200000122BL_B1/SS200000122BL_B1.gef --lasso-geojson /storeData/USER/data/01.CellBin/00.user/liyumei/tmp/SS200000122BL_B1_20250427150337.lasso.geojson --output /storeData/USER/data/01.CellBin/00.user/liyumei/tmp/register
```



### 五. **Reconnecting to stereo3d**

1. Manually modify file naming (to comply with stereo3d input format)
2. Manually modify directory structure (to comply with stereo3d input format)
3. Manually modify the record sheet to clarify preceding and subsequent relationships
4. Ensure the naming in the ssDNA_ChipNo column of the record sheet is consistent with the mask file naming
5. Run stereo3d to output results