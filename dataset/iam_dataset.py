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
            for xml_file in xml_files:
                file_path = os.path.join(xml_path, xml_file)
                file_samples = self._parse_single_xml(file_path)
                samples.extend(file_samples)
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
            
            # IAM XML structure: <form> -> <handwritten-part> -> <line>
            # or sometimes just <form> -> <line>
            for line in root.iter('line'):
                line_id = line.get('id')
                text = line.get('text')
                
                if line_id and text and text.strip():  # Ensure we have both id and non-empty text
                    # IAM line IDs are like: "a01-000u-00-00"
                    # Format: {writer}-{form}-{line_number}-{word_number}
                    parts = line_id.split('-')
                    if len(parts) >= 3:
                        writer_id = parts[0]  # e.g., "a01"
                        form_id = parts[1]    # e.g., "000u"
                        
                        # Image path in IAM dataset: lines/{writer_id}/{line_id}.png
                        image_path = f"{writer_id}/{line_id}.png"
                        
                        # Check if the image file actually exists
                        full_img_path = os.path.join(self.root_dir, image_path)
                        if os.path.exists(full_img_path):
                            samples.append((image_path, text.strip()))
                        else:
                            # Try alternative path structure if needed
                            alt_image_path = f"{writer_id}-{form_id}/{line_id}.png"
                            alt_full_path = os.path.join(self.root_dir, alt_image_path)
                            if os.path.exists(alt_full_path):
                                samples.append((alt_image_path, text.strip()))
            
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
