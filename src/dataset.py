import os
import glob
import hashlib
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class DentalAgeDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

        # Extract integer age from filename: "17 47.jpg" -> age = 17.0
        self.ages = []
        for fp in file_paths:
            basename = os.path.basename(fp)
            age = float(basename.split(" ")[0])
            self.ages.append(age)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        age = self.ages[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)


def resolve_data_dir(data_dir, use_hybrid=False):
    if use_hybrid:
        base_dir = os.path.dirname(data_dir)
        hybrid_dir = os.path.join(base_dir, "hybrid_dataset")
        if os.path.exists(hybrid_dir):
            data_dir = hybrid_dir
    return data_dir


def check_sha256_duplicates(file_paths):
    hash_map = defaultdict(list)
    for fp in sorted(file_paths):
        with open(fp, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        hash_map[digest].append(os.path.basename(fp))
    duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}
    return len(file_paths), len(hash_map), duplicates


def split_dataset_files(valid_files, random_state=42):
    sorted_files = sorted(valid_files)
    train_files, temp_files = train_test_split(
        sorted_files, test_size=0.30, random_state=random_state
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=random_state
    )
    return train_files, val_files, test_files


def get_dataloaders(
    data_dir,
    batch_size=8,
    img_size=(224, 448),
    use_hybrid=False,
    include_test=False,
    random_state=42,
    num_workers=0,
):
    resolved_dir = resolve_data_dir(data_dir, use_hybrid=use_hybrid)
    all_files = sorted(glob.glob(os.path.join(resolved_dir, "*.jpg")))
    valid_files = [f for f in all_files if " " in os.path.basename(f)]

    # Deterministic 3-way split: 70% train / 15% val / 15% independent test
    train_files, val_files, test_files = split_dataset_files(
        valid_files, random_state=random_state
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset = DentalAgeDataset(train_files, transform=train_transform)
    val_dataset = DentalAgeDataset(val_files, transform=eval_transform)
    test_dataset = DentalAgeDataset(test_files, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if include_test:
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def get_three_way_dataloaders(
    data_dir,
    batch_size=8,
    img_size=(224, 448),
    use_hybrid=False,
    random_state=42,
    num_workers=0,
):
    return get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        use_hybrid=use_hybrid,
        include_test=True,
        random_state=random_state,
        num_workers=num_workers,
    )

