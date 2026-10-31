import os
import json
import random

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")
ann_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
output_split_path = os.path.join(coco_dir, "frozen_splits.json")

# 2. Load the original captions file
print("📄 Reading captions file captions_train2017.json...")
with open(ann_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# Extract all unique image IDs
all_images = coco_data['images']
all_image_ids = list(set(img['id'] for img in all_images))
print(f"📊 Total available images in COCO train2017: {len(all_image_ids)}")

# 3. Fix the random seed for freezing and ensuring deterministic results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Shuffle IDs randomly after fixing the seed
random.shuffle(all_image_ids)

# 4. Extract frozen splits in fixed blocks
train_ids = all_image_ids[:10000]       # 10,000 images for the knowledge base (Train)
val_ids = all_image_ids[10000:10100]    # 100 images for Validation
test_ids = all_image_ids[10100:10200]   # 100 images for Final Testing

# 5. Programmatic Anti-Leakage Audit
intersection_train_test = set(train_ids).intersection(set(test_ids))
intersection_val_test = set(val_ids).intersection(set(test_ids))

assert len(intersection_train_test) == 0, "❌ ERROR: Data leakage detected between train and test sets!"
assert len(intersection_val_test) == 0, "❌ ERROR: Data leakage detected between validation and test sets!"

print("\n🔒 Anti-Leakage Audit passed: 0% intersection.")

# 6. Save the frozen splits to a JSON file
splits_data = {
    "seed": RANDOM_SEED,
    "train_ids": train_ids,
    "val_ids": val_ids,
    "test_ids": test_ids,
    "metadata": {
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids)
    }
}

with open(output_split_path, 'w', encoding='utf-8') as f:
    json.dump(splits_data, f, indent=4)

print(f"✅ Frozen splits successfully saved to: {output_split_path}")