## sheet file
__Note:__ Record table naming needs to follow the specification, like ```{project_name}_slice_records_{date}.xlsx```, eg
```E-ST20220923002_slice_records_20221110.xlsx```.
The file has two tables. Among them, Table ```Meta``` records the key original data, which will be used in the process of generating the mesh file; Table ```SliceSequence``` describes how any section of the three-dimensional tissue is utilized throughout the sectioning process. This information is critical to the entire process.

### Meta
|  Name   | Description  |
|  ----  | ----  |
| SampleName  | Tags for documenting organization |
| Magnification  | Optical imaging magnification |
| SizePerPixel  | The physical size of each pixel |
| CameraTravelDistance  | The movement distance of the camera in the X direction |
| Z-interval  | Robotic arm Z-axis stepping distance (critical for 3D mesh spacing) |

### SliceSequence
|  Name   | Description                                            | Value  |
|  ----  |--------------------------------------------------------| ----  |
| Slice_ID  | The slicing order of each slice                        | - |
| Z_index  | The cumulative distance of each slice along the Z-axis (in μm).For example: if the first slice is 5 μm thick, fill in 5; if the second slice is 10 μm thick, fill in 15 (5+10). This determines the vertical position of slices in the 3D model. | - |
| Idling  | Discard this slice                                     | - |
| SSDNA_SN  | The slicing order of the chip                          |  -|
| SSDNA_ChipNo  | The chip number of the spatiotemporal sequencing chip  | - |
| HE_SN  | The chip number of the spatiotemporal sequencing chip  | - |
