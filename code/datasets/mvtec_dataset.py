import os
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


from .synthetic_anomalies import patch_ex, patch_ex_guided

WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
                    'zipper':((0.03, 0.4), (0.03, 0.2)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}


NUM_PATCHES = {'bottle':3, 'cable':3, 'capsule':3, 'hazelnut':3, 'metal_nut':3, 
               'pill':3, 'screw':4, 'toothbrush':3, 'transistor':3, 'zipper':4,
               'carpet':4, 'grid':4, 'leather':4, 'tile':4, 'wood':4}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

# brightness, threshold pairs
BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}

OBJECTS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']


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


class MVtecDatasetGuided(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.target_size = (448, 448)
        # self.transform = transform
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

        self.mask_root_dir = _resolve_mask_root(self.root_dir, 'mvtec_fg')

        self.paths = []
        self.x = []
        self.mask_paths = []
        self.guided_masks = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and 'png' in file:
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))
                    mask_path = self._mask_path(file_path)
                    self.mask_paths.append(mask_path)
                    self.guided_masks.append(self._load_mask(mask_path))

        self.prev_idx = np.random.randint(len(self.paths))
        print("shape of guided masks:", self.guided_masks[0].shape)
        print("shape of first image:", np.asarray(self.x[0]).shape)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = self._class_name_from_path(img_path)
        guided_mask = self.guided_masks[index]

        self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
                    'num_patches': NUM_PATCHES.get(class_name),
                    'gamma_params':(2, 0.05, 0.03), 'resize':True,
                    'same':False,
                    'mode':cv2.NORMAL_CLONE,
                    'label_mode':'logistic-intensity'}
        if class_name in TEXTURES:
            self_sup_args['resize_bounds'] = (.5, 2)

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

    def _class_name_from_path(self, image_path: str) -> str:
        parts = os.path.normpath(image_path).split(os.sep)
        if len(parts) < 4:
            raise ValueError(f"Unexpected image path structure: {image_path}")
        return parts[-4]

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



    def collate(self, instances):

        images = []
        class_names = []
        masks = []
        img_paths = []
        for origin, anomaly, cls_name, mask, paths in instances:

            images.append(origin)
            class_names.append(cls_name)
            masks.append(torch.zeros_like(mask))
            img_paths.append(paths)

            images.append(anomaly)
            class_names.append(cls_name)
            masks.append(mask)
            img_paths.append(paths)

        return dict(
            images=images,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )


class MVtecDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # self.transform = transform
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

        self.paths = []
        self.x = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if "train" in file_path and "good" in file_path and 'png' in file:
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))

        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = img_path.split('/')[-4]

        self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
                    'num_patches': 2, #if single_patch else NUM_PATCHES.get(class_name),
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE,
                    'label_mode':'logistic-intensity',
                    'skip_background': BACKGROUND.get(class_name)}
        if class_name in TEXTURES:
            self_sup_args['resize_bounds'] = (.5, 2)

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
        for origin, anomaly, cls_name, mask, paths in instances:

            images.append(origin)
            class_names.append(cls_name)
            masks.append(torch.zeros_like(mask))
            img_paths.append(paths)

            images.append(anomaly)
            class_names.append(cls_name)
            masks.append(mask)
            img_paths.append(paths)

        return dict(
            images=images,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )
