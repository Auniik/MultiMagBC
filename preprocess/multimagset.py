import os
import random
from torch.utils.data import Dataset
from PIL import Image



class MultiMagPatientDataset(Dataset):
    def __init__(self, patient_dict, patient_ids, mags=['40', '100', '200', '400'], transform=None):
        self.patient_dict = patient_dict
        self.patient_ids = patient_ids
        self.mags = mags
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        entry = self.patient_dict[pid]
        images_dict = {}
        label = entry['label']
        # group images by mag
        mag_to_files = {mag: [] for mag in self.mags}
        for fpath in entry['images']:
            mag = os.path.basename(os.path.dirname(fpath)).replace('X', '')
            if mag in mag_to_files:
                mag_to_files[mag].append(fpath)
        # sample one image per mag
        for mag in self.mags:
            files = mag_to_files[mag]
            if not files:
                # fallback: random from any
                files = entry['images']
            img_path = random.choice(files)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images_dict[f'mag_{mag}'] = img
        return images_dict, label