import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class IAMDataset(Dataset):
    def __init__(self, root_dir, xml_path, samples=None, transform=None):
        self.root_dir = root_dir
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self.parse_xml(xml_path)
        print(f"Loaded {len(self.samples)} samples from dataset")
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, None)),  # normalize height
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def parse_xml(self, xml_path):
        samples = []
        print(f"Parsing XML from: {xml_path}")
        
        # If xml_path is a directory, parse all XML files in it
        if os.path.isdir(xml_path):
            xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
            print(f"Found {len(xml_files)} XML files: {xml_files[:5]}...")  # Show first 5
            
            # Debug: Parse first file to understand structure
            if xml_files:
                first_file = os.path.join(xml_path, xml_files[0])
                print(f"Debugging first XML file: {first_file}")
                self._debug_xml_structure(first_file)
            
            for xml_file in xml_files[:10]:  # Limit to first 10 files for debugging
                file_path = os.path.join(xml_path, xml_file)
                file_samples = self._parse_single_xml(file_path)
                print(f"Parsed {len(file_samples)} samples from {xml_file}")
                samples.extend(file_samples)
                
                # Stop if we have enough samples for testing
                if len(samples) >= 100:
                    break
        else:
            # If it's a single file
            samples = self._parse_single_xml(xml_path)
        
        print(f"Total samples parsed: {len(samples)}")
        return samples

    def _debug_xml_structure(self, xml_file_path):
        """Debug function to understand XML structure"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            print(f"Root tag: {root.tag}")
            print(f"Root attributes: {root.attrib}")
            
            # Print first few children
            for i, child in enumerate(root):
                if i < 3:  # Only show first 3 children
                    print(f"Child {i}: tag={child.tag}, attrib={child.attrib}")
                    # Check for nested elements
                    for j, grandchild in enumerate(child):
                        if j < 2:  # Only show first 2 grandchildren
                            print(f"  Grandchild {j}: tag={grandchild.tag}, attrib={grandchild.attrib}")
                            # Check for line elements
                            for k, ggchild in enumerate(grandchild):
                                if k < 2:
                                    print(f"    GGChild {k}: tag={ggchild.tag}, attrib={ggchild.attrib}")
        except Exception as e:
            print(f"Error debugging XML structure: {e}")

    def _parse_single_xml(self, xml_file_path):
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            samples = []
            
            # The IAM dataset has multiple possible XML structures:
            # 1. <form> -> <handwritten-part> -> <line>
            # 2. <form> -> <line>
            # 3. Direct <line> elements
            
            lines_found = 0
            valid_samples = 0
            
            # Try to find all 'line' elements regardless of nesting
            for line in root.iter('line'):
                lines_found += 1
                line_id = line.get('id')
                text = line.get('text')
                
                if line_id and text and text.strip():
                    # IAM line IDs are typically like: "a01-000u-00-00" or similar patterns
                    parts = line_id.split('-')
                    if len(parts) >= 2:
                        writer_id = parts[0]  # e.g., "a01"
                        
                        # Try multiple image path patterns
                        possible_paths = [
                            f"{writer_id}/{line_id}.png",           # Pattern 1: a01/a01-000u-00-00.png
                            f"{line_id}.png",                       # Pattern 2: a01-000u-00-00.png
                            f"{writer_id}-{parts[1]}/{line_id}.png" # Pattern 3: a01-000u/a01-000u-00-00.png
                        ]
                        
                        for image_path in possible_paths:
                            full_img_path = os.path.join(self.root_dir, image_path)
                            if os.path.exists(full_img_path):
                                samples.append((image_path, text.strip()))
                                valid_samples += 1
                                break
                        else:
                            # If no image found, still add the sample for debugging
                            if valid_samples < 5:  # Only show first 5 missing images
                                print(f"Image not found for line {line_id}, tried paths: {possible_paths}")
            
            if lines_found > 0:
                print(f"  Found {lines_found} line elements, {valid_samples} with existing images")
            
            return samples
            
        except ET.ParseError as e:
            print(f"Error parsing {xml_file_path}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error parsing {xml_file_path}: {e}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        
        try:
            img = Image.open(full_img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            # Return a dummy image and label
            dummy_img = Image.new('RGB', (200, 40), color='white')
            img = self.transform(dummy_img)
            return img, ""
