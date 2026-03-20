from dataclasses import dataclass


@dataclass
class RSNAPaths:
    image_dir: str = "/data/hxly/datasets/BoneAge/RSNA/rsna_all"
    train_csv: str = "/data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Training.csv"
    val_csv: str = "/data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Boneage_Validation.csv"
    test_csv: str = "/data/hxly/datasets/BoneAge/RSNA/annotations/RSNA_Annotations/RSNA_Gender_Testing.csv"
