import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class IAMDataset(Dataset):
    def __init__(self, root_dir, xml_path, transform=None):
        self.root_dir = root_dir
        self.samples = self.parse_xml(xml_path)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, None)),  # normalize height
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        samples = []
        for line in root.iter('line'):
            image_path = line.attrib['image'].replace('line', 'lines')
            text = line.attrib['text']
            samples.append((image_path, text))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        img = self.transform(img)
        return img, label
