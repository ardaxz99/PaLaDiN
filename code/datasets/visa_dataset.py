import os
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

from .synthetic_anomalies import patch_ex, patch_ex_guided



CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def _resolve_mask_root(dataset_root: str, mask_dir_name: str) -> str:
    dataset_root_path = Path(dataset_root).resolve()
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "fg_masks" / mask_dir_name,
        dataset_root_path.parent / mask_dir_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Expected guided masks under one of: {expected}")

WIDTH_BOUNDS_PCT = {
    'candle': ((0.03, 0.20), (0.03, 0.20)),
    'capsules': ((0.03, 0.20), (0.03, 0.20)),
    'cashew': ((0.03, 0.30), (0.03, 0.30)),
    'chewinggum': ((0.03, 0.30), (0.03, 0.30)),
    'fryum': ((0.03, 0.30), (0.03, 0.30)),
    'macaroni1': ((0.02, 0.20), (0.02, 0.20)),
    'macaroni2': ((0.02, 0.20), (0.02, 0.20)),
    'pcb1': ((0.03, 0.30), (0.03, 0.30)),
    'pcb2': ((0.03, 0.30), (0.03, 0.30)),
    'pcb3': ((0.03, 0.30), (0.03, 0.30)),
    'pcb4': ((0.03, 0.30), (0.03, 0.30)),
    'pipe_fryum': ((0.03, 0.25), (0.03, 0.25)),
}

INTENSITY_LOGISTIC_PARAMS = {
    'candle': (1/8, 16),
    'capsules': (1/4, 8),
    'cashew': (1/6, 12),
    'chewinggum': (1/6, 12),
    'fryum': (1/6, 12),
    'macaroni1': (1/4, 8),
    'macaroni2': (1/4, 8),
    'pcb1': (1/12, 24),
    'pcb2': (1/12, 24),
    'pcb3': (1/12, 24),
    'pcb4': (1/12, 24),
    'pipe_fryum': (1/6, 12),
}

NUM_PATCHES = {
    'candle': 2,
    'capsules': 3,
    'cashew': 3,
    'chewinggum': 3,
    'fryum': 3,
    'macaroni1': 3,
    'macaroni2': 3,
    'pcb1': 2,
    'pcb2': 2,
    'pcb3': 2,
    'pcb4': 2,
    'pipe_fryum': 3,
}


class VisaDatasetGuided(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.target_size = (448, 448)
        self.transform = transforms.Resize(
                                self.target_size, interpolation=transforms.InterpolationMode.BICUBIC
                            )
        self.mask_transform = transforms.Resize(
                                self.target_size, interpolation=transforms.InterpolationMode.NEAREST
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225),
                                ),
                            ]
                        )

        datas_csv_path = os.path.join(self.root_dir, "split_csv", "1cls.csv")

        self.mask_root_dir = _resolve_mask_root(self.root_dir, 'visa_fg')

        self.paths = []
        self.x = []
        self.mask_paths = []
        self.guided_masks = []

        with open(datas_csv_path, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if row[1] == 'train' and row[0] in CLASS_NAMES:
                    file_path = os.path.join(root_dir, row[3])
                    mask_path = self._mask_path(file_path)
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))
                    self.mask_paths.append(mask_path)
                    self.guided_masks.append(self._load_mask(mask_path))

        
        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = self._class_name_from_path(img_path)
        guided_mask = self.guided_masks[index]

        self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name, ((0.03, 0.2), (0.03, 0.2))),
                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name, (1/12, 24)),
                    'num_patches': NUM_PATCHES.get(class_name, 2),
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE, 
                    'label_mode':'logistic-intensity',
                    'resize_bounds': (.5, 2)
                    }

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)    
        x, mask, centers = patch_ex_guided(x, guided_mask, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index
        

        origin = self.norm_transform(origin)
        x = self.norm_transform(x)

   
        return origin, x, class_name, mask, img_path



    def collate(self, instances):

        images = []
        class_names = []
        masks = []
        img_paths = []
        for origin, anomaly, cls_name, mask, path in instances:
            images.append(origin)
            class_names.append(cls_name)
            masks.append(torch.zeros_like(mask))
            img_paths.append(path)

            images.append(anomaly)
            class_names.append(cls_name)
            masks.append(mask)
            img_paths.append(path)


        return dict(
            images=images,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )

    def _class_name_from_path(self, image_path: str) -> str:
        parts = os.path.normpath(image_path).split(os.sep)
        if len(parts) < 5:
            raise ValueError(f"Unexpected image path structure: {image_path}")
        return parts[-5]

    def _mask_path(self, image_path: str) -> str:
        class_name = self._class_name_from_path(image_path)
        file_name = os.path.basename(image_path)
        return os.path.join(self.mask_root_dir, class_name, file_name)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Guided mask not found for {mask_path}")
        with Image.open(mask_path) as mask_img:
            mask_resized = self.mask_transform(mask_img.convert('L'))
        mask_array = np.asarray(mask_resized)
        return (mask_array > 0).astype(np.uint8)


class VisaDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.transform = transforms.Resize(
                                (448, 448), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225),
                                ),
                            ]
                        )

        datas_csv_path = os.path.join(self.root_dir, "split_csv", "1cls.csv")

        self.paths = []
        self.x = []

        with open(datas_csv_path, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                if row[1] == 'train' and row[0] in CLASS_NAMES:
                    file_path = os.path.join(root_dir, row[3])
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))

        
        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = img_path.split('/')[-5]

        self_sup_args={'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                    'intensity_logistic_params': (1/12, 24),
                    'num_patches': 2,
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE, 
                    'label_mode':'logistic-intensity',
                    'skip_background': None,
                    'resize_bounds': (.5, 2)
                    }

        x = np.asarray(x)
        origin = x

        p = self.x[self.prev_idx]
        if self.transform is not None:
            p = self.transform(p)
        p = np.asarray(p)    
        x, mask, centers = patch_ex(x, p, **self_sup_args)
        mask = torch.tensor(mask[None, ..., 0]).float()
        self.prev_idx = index
        

        origin = self.norm_transform(origin)
        x = self.norm_transform(x)

   
        return origin, x, class_name, mask, img_path



    def collate(self, instances):

        images = []
        class_names = []
        masks = []
        img_paths = []
        for origin, anomaly, cls_name, mask, path in instances:
            images.append(origin)
            class_names.append(cls_name)
            masks.append(torch.zeros_like(mask))
            img_paths.append(path)

            images.append(anomaly)
            class_names.append(cls_name)
            masks.append(mask)
            img_paths.append(path)


        return dict(
            images=images,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )
