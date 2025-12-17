## Slices Record Sheet
__Note:__ Record table naming needs to follow the specification, like ```{project_name}_slice_records_{date}.xlsx```, eg
```E-ST20220923002_slice_records_20221110.xlsx```.
The file has two tables. Among them, Table ```Meta``` records the key original data, which will be used in the process of generating the mesh file; Table ```SliceSequence``` describes how any section of the three-dimensional tissue is utilized throughout the sectioning process. This information is critical to the entire process.

### Meta
|  Name   | Description  | Required |
|  ----  | ----  | ---|
| SampleName  | Tags for documenting organization | Optional |
| Magnification  | Optical imaging magnification | Optional |
| SizePerPixel  | The physical size of each pixel (in mm) |Required |
| CameraTravelDistance  | The movement distance of the camera in the X direction | Optional |
| Z-interval  | Larger values result in smoother meshes with fewer details, while smaller values relative to meshes with sharper angles and richer features (Recommend 0.008mm ) | Required |

### SliceSequence
|  Name   | Description            | Note  |Required |
|  ----  |-------------------------| ----  |----  |
| Slice_ID  | Sequential order of each tissue slice.       | - |Required |
| Z_index  | Cumulative Z-axis distance from the origin to the current slice (in μm). | **Example**: Slice 1 thickness = 5 μm → 5; Slice 2 thickness = 10 μm → 15 (5+10). |Required |
| Idling  | Flag to exclude this slice from 3D reconstruction.    | Mark as 1 to discard; 0 to include. |Optional |
| SSDNA_SN  | The slicing order of the chips   |  -|Optional |
| SSDNA_ChipNo  | Unique physical identifier (chip number) of the individual SSDNA chip.  | - |Required |
| HE_SN  | The chip number of the corresponding H&E stained image  | - |Optional |
