import os

# ==== CHANGE THESE ====
root = "/Users/surya/Desktop/Siemens Energy"
images_path = os.path.join(root, "/Users/surya/Desktop/Siemens Energy/welding defects image")
labels_path = os.path.join(root, "/Users/surya/Desktop/Siemens Energy/labels_siemens-energy-annotation_2026-01-29-06-03-19")
# =====================

image_exts = (".jpg", ".jpeg", ".png")

# Safety checks
if not os.path.exists(images_path):
    print("❌ Images folder not found")
    exit()

if not os.path.exists(labels_path):
    print("❌ Labels folder not found")
    exit()

# Collect files
images = [f for f in os.listdir(images_path) if f.lower().endswith(image_exts)]
labels = [f for f in os.listdir(labels_path) if f.lower().endswith(".txt")]

image_bases = {os.path.splitext(f)[0] for f in images}
label_bases = {os.path.splitext(f)[0] for f in labels}

# Missing & extra
missing_labels = image_bases - label_bases
extra_labels = label_bases - image_bases

# =====================
# RESULTS
# =====================
print("📊 DATASET SUMMARY")
print("------------------")
print(f"🖼️  Total images : {len(images)}")
print(f"🏷️  Total labels : {len(labels)}")

print(f"\n❌ Images missing labels : {len(missing_labels)}")
for img in sorted(missing_labels):
    print(" -", img)

print(f"\n❌ Labels without images : {len(extra_labels)}")
for lbl in sorted(extra_labels):
    print(" -", lbl)

if not missing_labels and not extra_labels:
    print("\n✅ Dataset is perfectly aligned!")
