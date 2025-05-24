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
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, None)),  # normalize height
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def parse_xml(self, xml_path):
        samples = []
        
        # If xml_path is a directory, parse all XML files in it
        if os.path.isdir(xml_path):
            for xml_file in os.listdir(xml_path):
                if xml_file.endswith('.xml'):
                    file_path = os.path.join(xml_path, xml_file)
                    samples.extend(self._parse_single_xml(file_path))
        else:
            # If it's a single file
            samples = self._parse_single_xml(xml_path)
        
        return samples

    def _parse_single_xml(self, xml_file_path):
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            samples = []
            for line in root.iter('line'):
                if 'text' in line.attrib:
                    # Build the image path from the line id
                    line_id = line.attrib['id']
                    # Extract form and line numbers from line_id (e.g., "a01-000u-00-00")
                    parts = line_id.split('-')
                    if len(parts) >= 4:
                        form_id = f"{parts[0]}-{parts[1]}"
                        image_path = f"{form_id}/{line_id}.png"
                        text = line.attrib['text']
                        samples.append((image_path, text))
            return samples
        except ET.ParseError as e:
            print(f"Error parsing {xml_file_path}: {e}")
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
            dummy_img = Image.new('RGB', (100, 40), color='white')
            img = self.transform(dummy_img)
            return img, ""
