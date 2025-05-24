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
        
        # Fixed transform - removed None parameter and improved error handling
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            # Custom resize that maintains aspect ratio with fixed height
            transforms.Lambda(self._resize_keep_aspect_ratio),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _resize_keep_aspect_ratio(self, img):
        """Resize image to height 40 while maintaining aspect ratio"""
        target_height = 40
        w, h = img.size
        if h == 0:  # Handle edge case
            return img
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        return img.resize((new_width, target_height), Image.LANCZOS)

    def parse_xml(self, xml_path):
        samples = []
        print(f"Parsing XML from: {xml_path}")
        
        # If xml_path is a directory, parse all XML files in it
        if os.path.isdir(xml_path):
            xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
            print(f"Found {len(xml_files)} XML files: {xml_files[:5]}...")  # Show first 5
            
            for xml_file in xml_files:  # Process all files, not just first 10
                file_path = os.path.join(xml_path, xml_file)
                file_samples = self._parse_single_xml(file_path)
                if file_samples:
                    print(f"Parsed {len(file_samples)} samples from {xml_file}")
                    samples.extend(file_samples)
                
                # Stop early for debugging if we have enough samples
                if len(samples) >= 1000:  # Increased limit for better training
                    print(f"Stopping early after collecting {len(samples)} samples for debugging")
                    break
        else:
            # If it's a single file
            samples = self._parse_single_xml(xml_path)
        
        print(f"Total samples parsed: {len(samples)}")
        return samples

    def _parse_single_xml(self, xml_file_path):
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            samples = []
            
            # Get form_id from the XML filename (e.g., a01-000u.xml -> a01-000u)
            xml_filename = os.path.basename(xml_file_path)
            form_id = xml_filename.replace('.xml', '')
            
            lines_found = 0
            valid_samples = 0
            
            # Find all 'line' elements in the handwritten-part
            for line in root.iter('line'):
                lines_found += 1
                line_id = line.get('id')
                text = line.get('text')
                
                if line_id and text and text.strip():
                    # IAM line IDs are like: "a01-000u-00" 
                    # Images are stored in: lines/a01/a01-000u/a01-000u-00.png
                    parts = line_id.split('-')
                    if len(parts) >= 3:
                        writer_id = parts[0]  # e.g., "a01"
                        form_part = parts[1]  # e.g., "000u"
                        
                        # Construct the correct image path based on IAM structure
                        # Path: writer_id/form_id/line_id.png
                        image_path = f"{writer_id}/{form_id}/{line_id}.png"
                        full_img_path = os.path.join(self.root_dir, image_path)
                        
                        if os.path.exists(full_img_path):
                            # Additional check: verify the image can be opened
                            try:
                                with Image.open(full_img_path) as test_img:
                                    # Check if image has valid dimensions
                                    if test_img.size[0] > 0 and test_img.size[1] > 0:
                                        samples.append((image_path, text.strip()))
                                        valid_samples += 1
                                    else:
                                        print(f"Invalid image dimensions: {full_img_path}")
                            except Exception as e:
                                print(f"Cannot open image {full_img_path}: {e}")
                        else:
                            # Debug first few missing images
                            if valid_samples < 3:
                                print(f"Image not found: {full_img_path}")
                                # Check what files actually exist in that directory
                                dir_path = os.path.dirname(full_img_path)
                                if os.path.exists(dir_path):
                                    files = os.listdir(dir_path)[:5]  # Show first 5 files
                                    print(f"  Directory exists, contains: {files}")
            
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
            
            # Verify image is valid before transforming
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {img.size}")
                
            img = self.transform(img)
            return img, label
            
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            
            # Create a proper dummy image with valid dimensions
            try:
                dummy_img = Image.new('RGB', (200, 40), color='white')
                
                # Add some text to the dummy image to make it more realistic
                from PIL import ImageDraw
                draw = ImageDraw.Draw(dummy_img)
                draw.text((10, 10), "DUMMY", fill='black')
                
                img = self.transform(dummy_img)
                return img, label if label else "dummy"
                
            except Exception as dummy_error:
                print(f"Error creating dummy image: {dummy_error}")
                # Last resort: return a minimal tensor
                import torch
                return torch.zeros(1, 40, 200), label if label else "dummy"
