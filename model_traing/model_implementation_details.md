# --- Dataset Validation Script ---

This script is designed to **validate the alignment between images and annotation labels** in the welding defect dataset. Ensuring that each image has a corresponding label and vice versa is crucial for **accurate model training**.

## Purpose
- Confirm that **all images have corresponding annotation files**.
- Detect **extra labels** that do not have associated images.
- Handle different image file formats (`.jpg`, `.jpeg`, `.png`) and ensure proper matching.
- Avoid potential training errors caused by **missing or mismatched data**.


## Script Workflow

1. **Define Paths**  
```python
root = "/Users/surya/Desktop/Siemens Energy"
images_path = os.path.join(root, "/Users/surya/Desktop/Siemens Energy/welding defects image")
labels_path = os.path.join(root, "/Users/surya/Desktop/Siemens Energy/labels_siemens-energy-annotation_2026-01-29-06-03-19")
```
- Set paths for images and labels.  
- All formats (`.jpg`, `.jpeg`, `.png`) are supported.

2. **Safety Checks**  
```python
if not os.path.exists(images_path):
    print(" Images folder not found")
    exit()

if not os.path.exists(labels_path):
    print(" Labels folder not found")
    exit()
```
- Ensures folders exist before processing.

3. **Collect Files & Extract Base Names**  
```python
images = [f for f in os.listdir(images_path) if f.lower().endswith(image_exts)]
labels = [f for f in os.listdir(labels_path) if f.lower().endswith(".txt")]

image_bases = {os.path.splitext(f)[0] for f in images}
label_bases = {os.path.splitext(f)[0] for f in labels}
```
- Extracts filenames without extensions to **match images with labels**.

4. **Identify Missing and Extra Files**  
```python
missing_labels = image_bases - label_bases
extra_labels = label_bases - image_bases
```
- `missing_labels`: images that **lack annotations**.  
- `extra_labels`: annotation files **without corresponding images**.

5. **Print Dataset Summary**  
```python
print(f" Total images : {len(images)}")
print(f" Total labels : {len(labels)}")
print(f" Images missing labels : {len(missing_labels)}")
print(f" Labels without images : {len(extra_labels)}")
```
- Provides a **clear overview** of dataset health.

6. **Final Check**  
```python
if not missing_labels and not extra_labels:
    print("\n Dataset is perfectly aligned!")
```
- Confirms **all images and labels are properly matched**.

---

##  Importance for Model Training

- **Accurate Annotation Matching:** YOLOv8 and other object detection models **require each image to have a corresponding annotation**. Missing or extra files can cause **training errors** or **incorrect model learning**.  
- **Support for Multiple Formats:** The dataset may include `.jpg`, `.jpeg`, and `.png` files. This script ensures **all are included and properly matched**.  
- **Error Prevention:** Detects potential **mismatches early**, preventing wasted compute time during training.  
- **Data Integrity:** Maintains **high-quality dataset hygiene**, which directly impacts **model performance and reliability**.

---

## Summary

This dataset validation script is a **critical pre-processing step** in your welding defect detection pipeline. It ensures **data consistency and integrity**, which are foundational for **reliable model training and accurate defect detection**.  

By running this script before training, engineers and researchers can **identify and fix misalignments**, ensuring that the AI system learns from **high-quality, correctly annotated data**.


-------------------------------------------------------------------------------------------------------------------------------


# --- Dataset Split & YAML Preparation ---

This script prepares the **welding defect dataset for YOLOv8 training** by creating proper **train, validation, and test splits**, and generating a **data.yaml file**. Proper data arrangement is critical for **model training, validation, and evaluation**.

##  Purpose

* Split the dataset into **train (80%), validation (15%), and test (5%)** sets.
* Ensure **images and labels are correctly matched**.
* Generate a **YOLOv8-compatible `data.yaml` file** for training.


##  Script Workflow

1. **Define Settings**

```python
root = "/Users/surya/Desktop/Siemens Energy"
output = "PPE_dataset_7"
allowed_exts = ('.jpg', '.jpeg', '.png')
SPLITS = {"train": 0.80, "val": 0.15, "test": 0.05}
```

* Specify dataset paths, allowed image extensions, and split percentages.

2. **Create Folder Structure**

```python
for split in SPLITS.keys():
    os.makedirs(os.path.join(output, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output, f"labels/{split}"), exist_ok=True)
```

* YOLOv8 requires separate folders for `images` and `labels` for each split.

3. **Collect Matching Image-Label Pairs**

* Only images with corresponding `.txt` annotation files are included.

4. **Random Shuffle & Split**

```python
random.shuffle(all_images)
train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]
```

* Ensures **balanced and randomized splits**.

5. **Copy Files to Split Folders**

```python
shutil.copy(os.path.join(images_path, img), os.path.join(output, f"images/{split_name}", img))
shutil.copy(os.path.join(labels_path, actual_lbl), os.path.join(output, f"labels/{split_name}", actual_lbl))
```

* Maintains correct **image-label alignment**.

6. **Generate `data.yaml`**

```python
dataset_yaml = {
    'path': os.path.abspath(output),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(class_names),
    'names': {i: name for i, name in enumerate(class_names)}
}
with open(os.path.join(output, "data.yaml"), 'w') as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)
```

* YAML file contains **dataset paths, number of classes, and class names** for YOLOv8 training.

---

##  Importance

* **Correct Splits:** Prevents **data leakage** between training and validation sets.
* **Proper YAML File:** YOLOv8 relies on `data.yaml` to locate images and labels for each split.
* **Training Accuracy:** Ensures **model learns correctly** from the intended data distribution.
* **Reproducibility:** Randomized and structured splits allow experiments to be **consistent and repeatable**.

---

##  Summary

This script is a **critical preprocessing step** that organizes the dataset and prepares it for **effective YOLOv8 training**. Correct splits and the YAML configuration are foundational for **accurate defect detection and reliable model performance**.


-------------------------------------------------------------------------------------------------------------------------------


## --- Model Training – YOLOv8 for Welding Defect Detection ---

This project uses **YOLOv8 (Ultralytics)** for detecting welding defects. The training is performed in **two phases**, with **phase 1 focusing on freezing the backbone** to preserve pre-trained features while adapting the classifier head to our dataset.

### **1. Environment & Settings**
- **Device:** `mps` (Apple Metal Performance Shaders for GPU acceleration)  
- **Model size:** Medium (`yolov8m.pt`)  
- **Input image size:** 640 × 640  
- **Batch size:** 8  
- **Epochs:** 50  
- **Learning rate:** 0.0005  
- **Dataset:** Specified via a YAML file (`data.yaml`)  
- **Project & Run Name:** Stored under `runs/train/ppe_yolov8_phase1_frozen`

---

### **2. Freezing the Backbone**
To prevent the pre-trained features of YOLOv8 from being modified during initial training (especially useful when the dataset is small), we **freeze the backbone layers**. Only the classification head is trained in this phase.  

```python
# Freeze the backbone layers (indices 0–7)
freeze=[0,1,2,3,4,5,6,7]
```

This approach:
- Protects pre-trained knowledge  
- Reduces overfitting  
- Speeds up convergence on small datasets  

---

### **3. Data Augmentation & Regularization**
Controlled augmentation is applied to improve model generalization:

| Augmentation | Value | Purpose |
|--------------|-------|---------|
| Mosaic       | 0.5   | Combines multiple images into one for diversity |
| Mixup        | 0.1   | Blends images to reduce overfitting |
| HSV Hue      | 0.015 | Adjust color hue slightly |
| HSV Saturation | 0.6 | Varies color intensity |
| HSV Value    | 0.4   | Varies brightness |
| Rotation     | 5°    | Slight rotation for robustness |
| Translation  | 0.05  | Minor positional shifts |
| Scale        | 0.5   | Random resizing |
| Flip LR      | 0.5   | Horizontal flip with 50% chance |

---

### **4. Training Execution**
The model is trained with early stopping to prevent overfitting if the validation loss does not improve for 15 epochs.  

```python
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8m.pt")

# Train model with frozen backbone
model.train(
    data="/Users/surya/Desktop/Siemens Energy/welding YAML File/data.yaml",
    imgsz=640,
    batch=8,
    epochs=50,
    lr0=0.0005,
    freeze=[0,1,2,3,4,5,6,7],
    patience=15,
    mosaic=0.5,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,
    degrees=5,
    translate=0.05,
    scale=0.5,
    fliplr=0.5,
    project="runs/train",
    name="ppe_yolov8_phase1_frozen",
    save=True,
    verbose=True
)
```

---

### **5. Saving Best Weights**
After training, the best-performing weights are saved for inference or future fine-tuning:

```python
import os

best_weights_phase1 = os.path.join(
    "runs/train",
    "ppe_yolov8_phase1_frozen",
    "weights",
    "best.pt"
)
print(f"Best weights saved at: {best_weights_phase1}")
```

> This ensures reproducibility. Phase 2 training will be conducted when additional data becomes available from Siemens Energy, allowing the model to be fully fine-tuned.

