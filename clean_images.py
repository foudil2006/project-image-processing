import os
import json
from PIL import Image
import imagehash

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")
splits_path = os.path.join(coco_dir, "frozen_splits.json")

# The path where the extracted images are located
images_base_dir = os.path.join(coco_dir, "images", "train2017")
output_cleaned_path = os.path.join(coco_dir, "cleaned_train_ids.json")

# 2. Load the frozen splits
print("📄 Loading frozen splits from frozen_splits.json...")
with open(splits_path, 'r', encoding='utf-8') as f:
    splits = json.load(f)

train_ids = splits["train_ids"]
print(f"📊 Initial Training Images: {len(train_ids)}")

# 3. Initialize variables for cleaning
seen_hashes = set()
cleaned_train_ids = []
corrupted_count = 0
duplicate_count = 0

print("🔍 Starting image cleaning and pHash duplicate detection...")
print("⏳ This might take a few minutes. Please wait...")

# 4. Iterate through each image ID in the training set
for img_id in train_ids:
    # COCO image filenames are exactly 12 digits long, padded with leading zeros
    filename = f"{str(img_id).zfill(12)}.jpg"
    img_path = os.path.join(images_base_dir, filename)
    
    # Check if the file physically exists
    if not os.path.exists(img_path):
        corrupted_count += 1
        continue

    # 5. Check for corruption and visual duplicates using pHash
    try:
        # Open the image using Pillow
        with Image.open(img_path) as img:
            # Calculate the Perceptual Hash (pHash)
            img_hash = imagehash.phash(img)
            
            # If the hash is already in our set, it's a visual duplicate
            if img_hash in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(img_hash)
                cleaned_train_ids.append(img_id)
                
    except Exception as e:
        # If Pillow fails to open the image, it is corrupted
        corrupted_count += 1

# 6. Summary and Saving results
print("\n✅ Cleaning Process Complete!")
print(f"📉 Corrupted/Missing images removed: {corrupted_count}")
print(f"👯 Duplicate images removed: {duplicate_count}")
print(f"🏆 Final Cleaned Training Images: {len(cleaned_train_ids)}")

# Save the new cleaned list of IDs for the BM25 indexing step
with open(output_cleaned_path, 'w', encoding='utf-8') as f:
    json.dump(cleaned_train_ids, f, indent=4)

print(f"📁 Cleaned IDs successfully saved to: {output_cleaned_path}")