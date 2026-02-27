import os
import random
import shutil
import yaml

# ==== SETTINGS ====
root = "/Users/surya/Desktop/Siemens Energy"
output = "PPE_dataset_7"
allowed_exts = ('.jpg', '.jpeg', '.png')

# The requested split percentages
SPLITS = {
    "train": 0.80,
    "val": 0.15,
    "test": 0.05
}

class_names = [
    "Lump defect", 
    "Spatter defect", 
    "Pin hole defect", 
    "Chips & Burr", 
    "Undercut defect", 
    "Welding protrusion"
]

images_path = "/Users/surya/Desktop/Siemens Energy/welding defects image"
labels_path = "/Users/surya/Desktop/Siemens Energy/labels_siemens-energy-annotation"
# ==========================

# Create YOLO folder structure for all three splits
for split in SPLITS.keys():
    os.makedirs(os.path.join(output, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output, f"labels/{split}"), exist_ok=True)

# Collect matching pairs
labels_lower = [f.lower() for f in os.listdir(labels_path) if f.lower().endswith(".txt")]
all_images = []

for img in os.listdir(images_path):
    if img.lower().endswith(allowed_exts):
        base = os.path.splitext(img)[0].lower()
        if (base + ".txt") in labels_lower:
            all_images.append(img)

random.shuffle(all_images)
total = len(all_images)

# Calculate split points
train_end = int(total * SPLITS["train"])
val_end = train_end + int(total * SPLITS["val"])

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

def copy_files(file_list, split_name):
    for img in file_list:
        base = os.path.splitext(img)[0].lower()
        actual_lbl = next((f for f in os.listdir(labels_path) if f.lower() == (base + ".txt")), None)
        if actual_lbl:
            shutil.copy(os.path.join(images_path, img), os.path.join(output, f"images/{split_name}", img))
            shutil.copy(os.path.join(labels_path, actual_lbl), os.path.join(output, f"labels/{split_name}", actual_lbl))

# Execute the copy
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

# Create YAML
names_dict = {i: name for i, name in enumerate(class_names)}
dataset_yaml = {
    'path': os.path.abspath(output),
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(class_names),
    'names': names_dict
}

with open(os.path.join(output, "data.yaml"), 'w') as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print(f" Created {len(train_files)} Train, {len(val_files)} Val, and {len(test_files)} Test files.")