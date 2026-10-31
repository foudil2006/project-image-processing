import os
import json
import pickle
import re
from rank_bm25 import BM25Okapi

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")
ann_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
cleaned_ids_path = os.path.join(coco_dir, "cleaned_train_ids.json")

# Output files for the index and the mapping
output_index_path = os.path.join(coco_dir, "bm25_index.pkl")
output_mapping_path = os.path.join(coco_dir, "bm25_mapping.json")

# 2. Load the cleaned image IDs and original captions
print("📄 Loading cleaned train IDs and captions...")
with open(cleaned_ids_path, 'r', encoding='utf-8') as f:
    # Use a set for O(1) fast lookup
    cleaned_train_ids = set(json.load(f)) 

with open(ann_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# 3. Extract and group captions by image ID
print("🔄 Extracting and mapping captions for cleaned images...")
image_captions = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id in cleaned_train_ids:
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(ann['caption'])

# 4. Text Cleaning and Tokenization function
def clean_and_tokenize(text):
    # Convert to lowercase
    text = text.lower()
    # Remove all non-alphanumeric characters (keep only letters, numbers, and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Split text into a list of words (tokens)
    tokens = text.split()
    return tokens

# 5. Prepare data for the BM25 model
print("⚙️ Cleaning text, tokenizing, and preparing the corpus...")
corpus_tokens = []
image_id_mapping = [] 

for img_id, captions in image_captions.items():
    # Combine all 5 captions into a single document per image
    combined_text = " ".join(captions)
    tokens = clean_and_tokenize(combined_text)
    
    corpus_tokens.append(tokens)
    image_id_mapping.append(img_id)

# 6. Initialize and build the BM25 model
print("🧠 Training BM25 model (Indexing)... This may take a few moments.")
bm25_model = BM25Okapi(corpus_tokens)

# 7. Save the model and mapping for the retrieval phase
print("💾 Saving BM25 index and mapping...")
with open(output_index_path, 'wb') as f:
    pickle.dump(bm25_model, f)

with open(output_mapping_path, 'w', encoding='utf-8') as f:
    json.dump(image_id_mapping, f, indent=4)

print("\n✅ BM25 Indexing Complete!")
print(f"📁 Index saved to: {output_index_path}")
print(f"📁 Mapping saved to: {output_mapping_path}")
print(f"🏆 Total indexed images (should match cleaned IDs): {len(image_id_mapping)}")