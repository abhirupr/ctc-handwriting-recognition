import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_iam_corpus(data_dir, output_file):
    """Extract all text labels from IAM dataset"""
    corpus_text = []
    data_path = Path(data_dir)
    
    print(f"Looking for data in: {data_path.absolute()}")
    
    # Method 1: Extract from XML files (IAM format)
    xml_files = list(data_path.glob("**/*.xml"))
    print(f"Found {len(xml_files)} XML files")
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract text from line elements in IAM XML format
            for line in root.findall(".//line"):
                text = line.get('text', '').strip()
                if text:
                    corpus_text.append(text)
            
            # Also try alternative XML structures
            for word in root.findall(".//word"):
                text = word.get('text', '').strip()
                if text:
                    corpus_text.append(text)
                    
        except Exception as e:
            print(f"Error processing XML {xml_file}: {e}")
            continue
    
    # Method 2: Extract from your IAMDataset directly
    try:
        # Import your dataset class
        import sys
        sys.path.append(str(data_path.parent))
        from dataset.iam_dataset import IAMDataset
        from config import DATA_DIR, XML_PATH
        
        print(f"Trying to load IAM dataset from {DATA_DIR}, {XML_PATH}")
        dataset = IAMDataset(DATA_DIR, XML_PATH)
        
        print(f"Found {len(dataset.samples)} samples in dataset")
        for sample in dataset.samples:
            if 'text' in sample:
                text = sample['text'].strip()
                if text:
                    corpus_text.append(text)
            elif 'transcription' in sample:
                text = sample['transcription'].strip()
                if text:
                    corpus_text.append(text)
                    
    except Exception as e:
        print(f"Could not load IAM dataset directly: {e}")
    
    # Method 3: Look for text files in common IAM locations
    text_patterns = [
        "**/*.txt",
        "**/words.txt", 
        "**/lines.txt",
        "**/sentences.txt",
        "**/ascii/**/*.txt",
        "**/ground_truth/**/*.txt"
    ]
    
    for pattern in text_patterns:
        text_files = list(data_path.glob(pattern))
        print(f"Pattern {pattern}: found {len(text_files)} files")
        
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Skip IAM metadata lines that start with '#'
                        if line.startswith('#'):
                            continue
                        
                        # IAM format: ID transcription
                        parts = line.strip().split(' ', 1)
                        if len(parts) >= 2:
                            text = parts[1].strip()
                            if text:
                                corpus_text.append(text)
                        elif len(parts) == 1 and parts[0]:
                            # Single line of text
                            text = parts[0].strip()
                            if text and not text.startswith('#'):
                                corpus_text.append(text)
                                
            except Exception as e:
                print(f"Error processing text file {text_file}: {e}")
                continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_corpus = []
    for text in corpus_text:
        if text not in seen:
            seen.add(text)
            unique_corpus.append(text)
    
    # Write to corpus file
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in unique_corpus:
            f.write(text + '\n')
    
    print(f"Extracted {len(unique_corpus)} unique text samples to {output_file}")
    return len(unique_corpus)

def inspect_data_structure(data_dir):
    """Inspect your data directory structure"""
    data_path = Path(data_dir)
    
    print(f"\n=== Data Directory Structure ===")
    print(f"Base path: {data_path.absolute()}")
    print(f"Exists: {data_path.exists()}")
    
    if data_path.exists():
        # Show directory structure
        print(f"\nDirectory contents:")
        for item in sorted(data_path.iterdir()):
            if item.is_dir():
                file_count = len(list(item.rglob("*")))
                print(f"  📁 {item.name}/ ({file_count} files)")
            else:
                print(f"  📄 {item.name}")
        
        # Look for specific file types
        file_types = {
            'XML files': list(data_path.glob("**/*.xml")),
            'Text files': list(data_path.glob("**/*.txt")),
            'JSON files': list(data_path.glob("**/*.json")),
            'CSV files': list(data_path.glob("**/*.csv")),
            'Image files': list(data_path.glob("**/*.png")) + list(data_path.glob("**/*.jpg"))
        }
        
        print(f"\nFile type breakdown:")
        for file_type, files in file_types.items():
            print(f"  {file_type}: {len(files)}")
            if files and len(files) <= 5:  # Show first few files if not too many
                for f in files[:3]:
                    print(f"    - {f.relative_to(data_path)}")
                if len(files) > 3:
                    print(f"    ... and {len(files)-3} more")

if __name__ == "__main__":
    # First inspect the data structure
    inspect_data_structure("data/")
    
    # Then extract corpus
    count = extract_iam_corpus("data/", "corpus.txt")
    
    if count == 0:
        print("\n❌ No text found. Please check the data structure output above.")
        print("Common IAM dataset locations:")
        print("  - data/ascii/lines.txt")
        print("  - data/ascii/words.txt") 
        print("  - data/xml/*.xml")
        print("  - data/ground_truth/*.txt")
    else:
        print(f"\n✅ Successfully extracted {count} text samples!")
        print("You can now build your language model with:")
        print("  cd ctcdecode/third_party/kenlm/build")
        print("  ./bin/lmplz -o 4 < ../../../corpus.txt > ../../../model.arpa")
        print("  ./bin/build_binary ../../../model.arpa ../../../model.binary")