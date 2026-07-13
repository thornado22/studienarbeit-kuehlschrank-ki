import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


TRANSFORM_REGISTRY = {
    # Pixelveränderungen & Rauschen
    "GaussNoise": A.GaussNoise,
    "SaltAndPepper": A.SaltAndPepper,
    "ISONoise": A.ISONoise,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "HueSaturationValue": A.HueSaturationValue,
    "GaussianBlur": A.GaussianBlur,
    "MotionBlur": A.MotionBlur,
    
    # Occlusion
    "Occlusion": A.CoarseDropout,
    
    # Drehen und Spiegeln
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "ShiftScaleRotate": A.ShiftScaleRotate,
    "Perspective": A.Perspective,
    "Affine": A.Affine

}


class AlbumentationsBuilder:
    def __init__(self, cfg:dict, image_size=(224, 224)):

        self.image_size = image_size
        self.transforms = [A.Resize(*image_size)]

        self._build_single(cfg.get("single", {}))
        self._build_groups(cfg.get("group", {}))

        # always append resize + normalize at the end
        self.transforms.extend([
            A.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            ),
            ToTensorV2()
        ])

        self.pipeline = A.Compose(self.transforms)


    def _clean_params(self, params):
        clean = {}
        for k, v in params.items():
            if isinstance(v, list):
                clean[k] = tuple(v)
            elif v == "True":
                clean[k] = True
            elif v == "False":
                clean[k] = False
            else:
                clean[k] = v
        return clean

    def _build_single(self, single_cfg):
        for name, params in single_cfg.items():
            p = params.get("p", 1)

            if p == 0:
                continue

            cls = TRANSFORM_REGISTRY[name]
            params = self._clean_params(params)

            transform = cls(**params)
            self.transforms.append(transform)

    def _build_groups(self, group_cfg):
        for _, group in group_cfg.items():

            group_p = group.get("p", 0.5)
            transforms_cfg = group.get("transforms", {})

            group_transforms = []

            for name, params in transforms_cfg.items():
                cls = TRANSFORM_REGISTRY[name]
                params = self._clean_params(params)

                params["p"] = 1.0

                group_transforms.append(cls(**params))

            if group_transforms:
                self.transforms.append(
                    A.OneOf(group_transforms, p=group_p)
                )
    


    

