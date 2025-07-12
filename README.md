# Automatic Segmentation of the Sphenoid Sinus from CT Scans

This document explains how to train a neural network to automatically segment the sphenoid sinus from CT images and generate a 3D model (STL). Two main approaches are described:

1. U-Net — works with full 3D volumes and provides high-accuracy volumetric segmentation.
2. YOLOv8 / YOLOv11 — performs fast 2D segmentation on individual CT slices and can be combined into a 3D output.

---

## U-Net for 3D CT Scan Segmentation

### What it does:

U-Net takes a full CT volume (multiple 2D slices stacked together) and outputs a corresponding volume where the region of interest (the sphenoid sinus) is marked as a 3D binary mask.

### Data you need:

* A folder of CT slice images (PNG or extracted from DICOM)
* For each patient or scan, you also need a set of matching mask images showing where the sinus is

### Preparing the data:

You must load and stack the 2D slices into a 3D array. Here’s how it might look in Python:

```python
import numpy as np
import cv2
import os

volume = []
for filename in sorted(os.listdir("ct_slices/")):
    img = cv2.imread(f"ct_slices/{filename}", cv2.IMREAD_GRAYSCALE)
    volume.append(img)
volume = np.stack(volume)  # shape: (depth, height, width)
```

Repeat the same process for the segmentation masks.

### Training the model:

To train a U-Net, use a framework like MONAI. It provides easy tools for medical imaging.
Install MONAI:

```bash
pip install monai
```

Define a 3D U-Net model:

```python
from monai.networks.nets import UNet
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
```

Train it using a Dice loss function and Adam optimizer.

```python
from monai.losses import DiceLoss
import torch

loss_fn = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Run training for multiple epochs, feeding in pairs of volumes and masks.

### After training:

Once the model is trained, you can pass a new CT scan volume through it to get a predicted mask. To create a 3D model file from this mask:

```python
from skimage import measure
import trimesh

verts, faces, _, _ = measure.marching_cubes(mask_array, level=0.5)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export("sphenoid_sinus.stl")
```

This STL file can then be viewed or used in 3D modeling software.

---

## YOLOv8 / YOLOv11 for Slice-by-Slice Segmentation

### What it does:

YOLO-based models work on 2D images and can detect or segment objects on each CT slice. In our case, we use it to detect the sinus region on each slice.

### Required data:

* CT slices in PNG or JPG format
* For each slice, a label file with polygon coordinates of the segmented region

### Data structure:

Images and labels must be organized like this:

```
data/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each label file should be named like its image (e.g., 0001.png → 0001.txt) and contain:

```
0 x1 y1 x2 y2 x3 y3 ... xn yn
```

This describes one polygon for class "0".

### Training YOLOv8:

Install the YOLO package:

```bash
pip install ultralytics
```

Create a config file (data.yaml):

```yaml
path: data
train: images/train
val: images/val
names: ["sphenoid"]
```

Run the training command:

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml epochs=100 imgsz=640
```

This starts training a segmentation model using the training images and labels.

### Running predictions:

Once trained, you can apply the model to test images:

```bash
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=images/test/
```

This will produce new images with segmented masks drawn on top.

### Convert to 3D:

If you want a full 3D model, combine the 2D masks from each slice:

```python
import numpy as np
import cv2
from skimage import measure
import trimesh

volume = np.stack([cv2.imread(f, 0) for f in sorted(mask_files)])
verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export("sphenoid_yolo.stl")
```

---

Both methods let you automatically segment the sphenoid sinus. U-Net offers higher precision and is better suited for full-volume medical imaging. YOLO is easier to deploy and very fast for real-time systems. In both cases, the result can be turned into a 3D STL model for further use.
