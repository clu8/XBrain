from torchvision import transforms
from PIL import Image


IMG_SIZE = 1024

preprocess = transforms.Compose([
    transforms.Scale(IMG_SIZE, interpolation=Image.ANTIALIAS),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor()
])
