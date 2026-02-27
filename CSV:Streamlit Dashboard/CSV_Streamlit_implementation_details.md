# Siemens Energy Welding Inspection Dashboards

This repository includes **two key components** for welding defect detection and FPY analytics:

1. **CSV Dashboard (CDB)** – structured logging and reporting system.
2. **Streamlit Dashboard (SD)** – interactive live inspection and visualization portal.

Each component is explained with **code snippets** and **professional documentation** for GitHub presentation.

---

## CSV Dashboard (CDB)

### Purpose

The CSV dashboard maintains a **structured record of all inspections**, enabling data-driven quality control and FPY (First Pass Yield) calculations.

### CSV Structure

| timestamp | image_name | defect_classes | defect_summary | total_defects |
| --------- | ---------- | -------------- | -------------- | ------------- |

### Code Explanation

```python
import csv
from datetime import datetime
import pandas as pd

CSV_PATH = 'outputs/results.csv'

# Initialize CSV
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["timestamp", "image_name", "defect_classes", "defect_summary", "total_defects"])
    df.to_csv(CSV_PATH, index=False)

# Append new inspection result
new_row = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "weld_01.jpg",
    "Lump defect, Spatter defect",
    "Lump defect(1), Spatter defect(2)",
    3
]

with open(CSV_PATH, "a", newline="") as f:
    csv.writer(f).writerow(new_row)
```

**Highlights:**

* Ensures **data integrity** and consistent logging.
* Facilitates **filtering, searching, and reporting**.
* Enables **KPIs and FPY calculations**.

### FPY Example

```python
df = pd.read_csv(CSV_PATH)
total_units = len(df)
defect_free_units = len(df[df['total_defects'] == 0])
fpy = (defect_free_units / total_units) * 100
print(f"First Pass Yield (FPY): {fpy:.2f}%")
```

**Impact:** Provides **reliable historical data** for defect analysis and production optimization.

-------------------------------------------------------------------------------------------------------------------------------

##  Streamlit Dashboard (SD)

### Purpose

The Streamlit dashboard provides a **live interactive portal** for batch inspection, visualization, and analytics.

### Key Features

* **Live Batch Inspection:** Drag-and-drop multiple images.
* **Real-Time Detection:** YOLOv8 inference with bounding boxes and class labels.
* **Filters & Settings:** Image name search, defect type filter, confidence slider.
* **Analytics:** KPI metrics, defect type bar/pie charts, historical image viewer.
* **Reporting:** Download batch results as CSV.

### Code Explanation

```python
import streamlit as st
from ultralytics import YOLO
import cv2, numpy as np
from collections import Counter
from datetime import datetime

# Load Model
MODEL_PATH = 'path/to/model.pt'
model = YOLO(MODEL_PATH)

# CSV initialization
CSV_PATH = 'outputs/results.csv'
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["timestamp", "image_name", "defect_classes", "defect_summary", "total_defects"])
    df.to_csv(CSV_PATH, index=False)

# Live Upload Portal
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png"], accept_multiple_files=True)
for file in uploaded_files:
    img_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    results = model(img)[0]
    detected_labels = [int(box.cls[0]) for box in results.boxes]
    # Save results to CSV
    # Update live Streamlit view
```

**Highlights:**

* Provides **real-time inspection feedback**.
* Visualizes **defect distributions** with **bar and pie charts**.
* Allows detailed **image-level inspection and history review**.
* Fully interactive for operators and engineers.

-------------------------------------------------------------------------------------------------------------------------------

## Summary

These two dashboards complement each other:

* **CSV Dashboard** ensures **structured data storage** for auditing and FPY metrics.
* **Streamlit Dashboard** provides **interactive inspection, visualization, and reporting**.

This setup ensures **industrial-grade quality control** and enables **data-driven production decisions**.
