import os
import json
import re
from collections import Counter
import matplotlib.pyplot as plt

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")

captions_ann_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
instances_ann_path = os.path.join(coco_dir, "annotations", "instances_train2017.json")

splits_path = os.path.join(coco_dir, "frozen_splits.json")
cleaned_ids_path = os.path.join(coco_dir, "cleaned_train_ids.json")

# Create output directory for plots
plots_dir = os.path.join(project_dir, "eda_plots")
os.makedirs(plots_dir, exist_ok=True)

# 2. Load data files
print("📄 Loading data for comprehensive EDA according to project requirements...")
with open(splits_path, 'r', encoding='utf-8') as f:
    splits = json.load(f)

with open(cleaned_ids_path, 'r', encoding='utf-8') as f:
    cleaned_train_ids = set(json.load(f))

with open(captions_ann_path, 'r', encoding='utf-8') as f:
    captions_data = json.load(f)

# Load instances for object class distribution & scene complexity if available
has_instances = os.path.exists(instances_ann_path)
if has_instances:
    print("📄 Loading COCO instances annotations...")
    with open(instances_ann_path, 'r', encoding='utf-8') as f:
        instances_data = json.load(f)

# ---------------------------------------------------------
# Plot 1: Dataset Splits Pie Chart
# ---------------------------------------------------------
print("📊 Plot 1: Dataset Splits Pie Chart...")
val_count = splits["metadata"]["num_val"]
test_count = splits["metadata"]["num_test"]
train_clean_count = len(cleaned_train_ids)
corrupted_duplicate_count = splits["metadata"]["num_train"] - train_clean_count

labels = ['Cleaned Train (KB)', 'Validation', 'Test', 'Removed (Dup/Corrupted)']
sizes = [train_clean_count, val_count, test_count, corrupted_duplicate_count]
colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=(0.05, 0, 0, 0), labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Dataset Splits After Anti-Leakage & Cleaning")
plt.savefig(os.path.join(plots_dir, "1_dataset_splits_pie.png"), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 2 & 3: Caption Lengths & Top Words Frequency
# ---------------------------------------------------------
print("📊 Plot 2 & 3: Analyzing Caption Lengths and Vocabulary...")
caption_lengths = []
all_words = []
stop_words = {"a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "at", "by", "for", "it", "are", "that"}

for ann in captions_data['annotations']:
    if ann['image_id'] in cleaned_train_ids:
        text = ann['caption'].lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        caption_lengths.append(len(words))
        for word in words:
            if word not in stop_words:
                all_words.append(word)

# Histogram of Caption Lengths
plt.figure(figsize=(10, 5))
plt.hist(caption_lengths, bins=range(min(caption_lengths), max(caption_lengths) + 1, 1), 
         color='#9C27B0', edgecolor='black', alpha=0.7)
plt.title("Caption Length Distribution (Words per Caption)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.4)
plt.savefig(os.path.join(plots_dir, "2_caption_length_hist.png"), dpi=300, bbox_inches='tight')
plt.close()

# Bar chart of Top 20 Words
word_counts = Counter(all_words)
top_20 = word_counts.most_common(20)
plt.figure(figsize=(12, 5))
plt.bar([x[0] for x in top_20], [x[1] for x in top_20], color='#00BCD4', edgecolor='black')
plt.title("Top 20 Frequent Words in Captions")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.4)
plt.savefig(os.path.join(plots_dir, "3_top_words_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Plot 4 & 5: Object Class Distribution & Scene Complexity (COCO Requirements)
# ---------------------------------------------------------
if has_instances:
    print("📊 Plot 4 & 5: Object Classes & Scene Complexity...")
    # Map category ID to category name
    cat_map = {cat['id']: cat['name'] for cat in instances_data['categories']}
    
    image_object_counts = Counter()
    class_counts = Counter()

    for ann in instances_data['annotations']:
        img_id = ann['image_id']
        if img_id in cleaned_train_ids:
            cat_name = cat_map[ann['category_id']]
            class_counts[cat_name] += 1
            image_object_counts[img_id] += 1

    # Top 30 Object Classes
    top_30_classes = class_counts.most_common(30)
    plt.figure(figsize=(14, 6))
    plt.bar([x[0] for x in top_30_classes], [x[1] for x in top_30_classes], color='#FF5722', edgecolor='black')
    plt.title("Top 30 COCO Object Classes Frequency")
    plt.xlabel("Object Class")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=60, ha='right')
    plt.grid(axis='y', alpha=0.4)
    plt.savefig(os.path.join(plots_dir, "4_top_object_classes.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Scene Complexity (Objects per Image)
    objects_per_img = [image_object_counts[img_id] for img_id in cleaned_train_ids]
    plt.figure(figsize=(10, 5))
    plt.hist(objects_per_img, bins=range(0, max(objects_per_img) + 1, 1), 
             color='#3F51B5', edgecolor='black', alpha=0.7)
    plt.title("Scene Complexity (Number of Objects per Image)")
    plt.xlabel("Number of Objects Detected")
    plt.ylabel("Number of Images")
    plt.grid(axis='y', alpha=0.4)
    plt.savefig(os.path.join(plots_dir, "5_scene_complexity.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("\n✅ Updated EDA complete! All required charts generated in: " + plots_dir)
