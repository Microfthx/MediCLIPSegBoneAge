from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RSNABoneAgePromptDataset(Dataset):
    def __init__(self, image_dir, csv_path, image_size=224, has_labels=True, train=True):
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)
        self.has_labels = has_labels
        self.annotations = pd.read_csv(self.csv_path)
        self.transform = self._build_transform(image_size=image_size, train=train)

    def _build_transform(self, image_size, train):
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomAffine(degrees=12, translate=(0.03, 0.03), scale=(0.97, 1.03)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        image_id = int(row["ID"])
        image_path = self.image_dir / f"{image_id}.png"

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        is_male = int(bool(row["Male"]))
        gender = torch.tensor([float(is_male)], dtype=torch.float32)
        sex_text = "male" if is_male else "female"
        prompt = f"left hand radiograph for bone age assessment of a {sex_text} pediatric patient"

        sample = {
            "image": image,
            "gender": gender,
            "prompt": prompt,
            "id": image_id,
        }

        if self.has_labels and "Boneage" in row.index:
            sample["target"] = torch.tensor(float(row["Boneage"]), dtype=torch.float32)

        return sample
