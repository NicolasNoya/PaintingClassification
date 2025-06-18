import random
from PIL import Image
from typing import Literal
from torchvision import transforms as T

class SkewTransform:
    def __init__(self, magnitude=(0.1, 0.4), direction="horizontal"):
        self.magnitude = magnitude
        self.direction = direction

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = T.ToPILImage()(img)

        w, h = img.size
        mag = random.uniform(*self.magnitude)

        if self.direction == "horizontal":
            matrix = (1, mag * w / h, 0, 0, 1, 0)
        else:
            matrix = (1, 0, 0, mag * h / w, 1, 0)

        img = img.transform(img.size, Image.AFFINE, matrix)
        return T.ToTensor()(img)

class RandomStretch:
    def __init__(self, side: Literal["width", "height"], stretch=(300, 450)):
        self.side = side
        self.stretch = stretch
        self.crop = T.CenterCrop(224)

    def __call__(self, img):
        val = random.randint(*self.stretch)
        size = (224, val) if self.side == "height" else (val, 224)
        return self.crop(T.Resize(size)(img))
    
    