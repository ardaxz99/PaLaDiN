from PIL import Image
import torch
from torchvision import transforms


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    data_transform = transforms.Compose(
        [
            transforms.Resize(
                (448, 448), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    image_outputs = []
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image_outputs.append(data_transform(image).to(device))

    return torch.stack(image_outputs, dim=0)
