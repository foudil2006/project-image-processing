import os
import json
import re
from collections import Counter
import matplotlib.pyplot as plt

# 1. Define static project paths
project_dir = r"C:\Users\THINKPAD\Desktop\projct image prossing"
coco_dir = os.path.join(project_dir, "datasets", "coco")
ann_path = os.path.join(coco_dir, "annotations", "captions_train2017.json")
splits_path = os.path.join(coco_dir, "frozen_splits.json")
cleaned_ids_path = os.path.join(coco_dir, "cleaned_train_ids.json")

# Create a directory to save the plots
plots_dir = os.path.join(project_dir, "eda_plots")
os.makedirs(plots_dir, exist_ok=True)

# 2. Load data files
print("📄 Loading data for Exploratory Data Analysis (EDA)...")
with open(splits_path, 'r', encoding='utf-8') as f:
    splits = json.load(f)

with open(cleaned_ids_path, 'r', encoding='utf-8') as f:
    cleaned_train_ids = set(json.load(f))

with open(ann_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# 3. Analyze Dataset Splits
val_count = splits["metadata"]["num_val"]
test_count = splits["metadata"]["num_test"]
train_clean_count = len(cleaned_train_ids)
corrupted_duplicate_count = splits["metadata"]["num_train"] - train_clean_count

print("📊 Generating Dataset Splits Pie Chart...")
labels = ['Cleaned Train (Database)', 'Validation', 'Test', 'Removed (Corrupted/Dup)']
sizes = [train_clean_count, val_count, test_count, corrupted_duplicate_count]
colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
explode = (0.05, 0, 0, 0)  

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Dataset Splits After Anti-Leakage and Cleaning")
plt.savefig(os.path.join(plots_dir, "dataset_splits_pie.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Analyze Captions (Text Length and Word Frequency)
print("🔄 Processing captions for the cleaned training set...")
caption_lengths = []
all_words = []

# Basic stop words to filter out common English words for better visualization
stop_words = {"a", "an", "the", "in", "on", "of", "and", "is", "to", "with", "at", "by", "for", "it"}

for ann in coco_data['annotations']:
    if ann['image_id'] in cleaned_train_ids:
        text = ann['caption'].lower()
        # Keep only alphanumeric characters and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        
        caption_lengths.append(len(words))
        
        # Add to word list if not a stop word
        for word in words:
            if word not in stop_words:
                all_words.append(word)

# 5. Generate Caption Length Histogram
print("📊 Generating Caption Length Distribution Histogram...")
plt.figure(figsize=(10, 6))
plt.hist(caption_lengths, bins=range(min(caption_lengths), max(caption_lengths) + 1, 1), 
         color='#9C27B0', edgecolor='black', alpha=0.7)
plt.title("Distribution of Caption Lengths (Word Count)")
plt.xlabel("Number of Words per Caption")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.5)
plt.savefig(os.path.join(plots_dir, "caption_length_hist.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. Generate Top Words Bar Chart
print("📊 Generating Top 20 Words Bar Chart...")
word_counts = Counter(all_words)
top_20_words = word_counts.most_common(20)

words = [item[0] for item in top_20_words]
counts = [item[1] for item in top_20_words]

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='#00BCD4', edgecolor='black')
plt.title("Top 20 Most Frequent Words in Cleaned Captions (Excluding Stop Words)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.5)
plt.savefig(os.path.join(plots_dir, "top_words_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ EDA Process Complete!")
print(f"📁 You can find your high-resolution charts ready for the report in: {plots_dir}")