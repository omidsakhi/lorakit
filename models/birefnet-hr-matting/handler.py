# These HF deployment codes refer to https://huggingface.co/not-lain/BiRefNet/raw/main/handler.py.
from typing import Dict, List, Any, Tuple
import os
import requests
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

torch.set_float32_matmul_precision(["high", "highest"][0])

device = "cuda" if torch.cuda.is_available() else "cpu"

### image_proc.py
def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


class ImagePreprocessor():
    def __init__(self, resolution: Tuple[int, int] = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image: Image.Image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-HR': 'BiRefNet_HR',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'General-reso_512': 'BiRefNet-reso_512',
    'Matting': 'BiRefNet-matting',
    'Matting-HR': 'BiRefNet_HR-Matting',
    'Portrait': 'BiRefNet-portrait',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy'
}

# Choose the version of BiRefNet here.
usage = 'General-HR'

# Set resolution
if usage in ['General-Lite-2K']:
    resolution = (2560, 1440)
elif usage in ['General-reso_512']:
    resolution = (512, 512)
elif usage in ['General-HR', 'Matting-HR']:
    resolution = (2048, 2048)
else:
    resolution = (1024, 1024) 

half_precision = True

class EndpointHandler():
    def __init__(self, path=''):
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            '/'.join(('zhengpeng7', usage_to_weights_file[usage])), trust_remote_code=True
        )
        self.birefnet.to(device)
        self.birefnet.eval()
        if half_precision:
            self.birefnet.half()

    def __call__(self, data: Dict[str, Any]):
        """
        data args:
            inputs (:obj: `str`)
            date (:obj: `str`)
        Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        print('data["inputs"] = ', data["inputs"])
        image_src = data["inputs"]
        if isinstance(image_src, str):
            if os.path.isfile(image_src):
                image_ori = Image.open(image_src)
            else:
                response = requests.get(image_src)
                image_data = BytesIO(response.content)
                image_ori = Image.open(image_data)
        else:
            image_ori = Image.fromarray(image_src)

        image = image_ori.convert('RGB')
        # Preprocess the image
        image_preprocessor = ImagePreprocessor(resolution=tuple(resolution))
        image_proc = image_preprocessor.proc(image)
        image_proc = image_proc.unsqueeze(0)

        # Prediction
        with torch.no_grad():
            preds = self.birefnet(image_proc.to(device).half() if half_precision else image_proc.to(device))[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        image_masked = refine_foreground(image, pred_pil)
        image_masked.putalpha(pred_pil.resize(image.size))
        return image_masked
