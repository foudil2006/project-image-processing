import os
import json
import pickle
import re

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")
ann_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
splits_path = os.path.join(coco_dir, "frozen_splits.json")

# Paths for the BM25 model and mapping
index_path = os.path.join(coco_dir, "bm25_index.pkl")
mapping_path = os.path.join(coco_dir, "bm25_mapping.json")

# 2. Load the necessary files
print("📂 Loading frozen splits, index, mapping, and annotations...")
with open(splits_path, 'r', encoding='utf-8') as f:
    test_ids = json.load(f)['test_ids']

with open(index_path, 'rb') as f:
    bm25_model = pickle.load(f)

with open(mapping_path, 'r', encoding='utf-8') as f:
    image_id_mapping = json.load(f)

with open(ann_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# 3. Select a test image and extract its caption to use as a query
# We will use the very first image in our frozen test set
target_test_id = test_ids[0]
user_query = ""

# Find the first caption associated with this test image
for ann in coco_data['annotations']:
    if ann['image_id'] == target_test_id:
        user_query = ann['caption']
        break

print("\n" + "="*50)
print(f"🎯 SIMULATING USER SEARCH")
print(f"Target Test Image ID: {target_test_id} (Not in Database due to Anti-Leakage)")
print(f"User Text Query: '{user_query}'")
print("="*50)

# 4. Clean and tokenize the query (Must use the exact same logic as indexing)
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

tokenized_query = clean_and_tokenize(user_query)
print(f"🔍 Tokenized Query: {tokenized_query}")

# 5. Search the BM25 Index
print("\n⚙️ Calculating scores using BM25 brain...")
# This returns a list of scores for all 10,000 images
scores = bm25_model.get_scores(tokenized_query)

# 6. Rank and retrieve the Top-K results
K = 5
# Pair each score with its internal index, then sort in descending order
indexed_scores = list(enumerate(scores))
indexed_scores.sort(key=lambda x: x[1], reverse=True)

# Extract the top K matches
top_k_results = indexed_scores[:K]

# 7. Map internal indices back to real COCO Image IDs
print(f"\n🏆 TOP {K} RETRIEVED IMAGES FROM KNOWLEDGE BASE:")
for rank, (internal_idx, score) in enumerate(top_k_results, 1):
    real_image_id = image_id_mapping[internal_idx]
    
    # Format the ID to match the COCO image filename standard
    filename = f"{str(real_image_id).zfill(12)}.jpg"
    
    print(f"Rank {rank}: Image ID {real_image_id} | Score: {score:.4f} | File: {filename}")

print("\n✅ Retrieval Test Complete!")
print("💡 Tip: You can go to the 'images/train2017' folder and view these Top 5 images to see if they match the query description visually!")