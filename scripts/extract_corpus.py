import os
import json
from pathlib import Path

def extract_corpus_from_dataset(data_dir, output_file):
    """Extract all text labels from your training dataset"""
    corpus_text = []
    
    # Look for annotation files in your data directory
    data_path = Path(data_dir)
    
    # Common patterns for handwriting datasets
    annotation_patterns = [
        "**/*.txt",      # Text files
        "**/*.json",     # JSON annotations
        "**/*.csv",      # CSV files
        "**/*.xml",      # XML files (IAM format)
    ]
    
    for pattern in annotation_patterns:
        for file_path in data_path.glob(pattern):
            try:
                if file_path.suffix == '.json':
                    # Handle JSON annotations
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'text' in data:
                            text = data['text'].strip()
                            if text:
                                corpus_text.append(text)
                        elif isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    text = item['text'].strip()
                                    if text:
                                        corpus_text.append(text)
                
                elif file_path.suffix == '.txt':
                    # Handle plain text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            text = line.strip()
                            if text and not text.startswith('#'):  # Skip comments
                                corpus_text.append(text)
                
                elif file_path.suffix == '.csv':
                    # Handle CSV files
                    import csv
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Look for common text column names
                            for col in ['text', 'transcription', 'label', 'ground_truth']:
                                if col in row and row[col]:
                                    text = row[col].strip()
                                    if text:
                                        corpus_text.append(text)
                                    break
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    # Write to corpus file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in corpus_text:
            f.write(text + '\n')
    
    print(f"Extracted {len(corpus_text)} text samples to {output_file}")
    return len(corpus_text)

if __name__ == "__main__":
    # Extract from your data directory
    count = extract_corpus_from_dataset("data/", "corpus.txt")
    if count == 0:
        print("No text found in data directory. Please check your data structure.")