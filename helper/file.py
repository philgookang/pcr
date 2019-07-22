from PIL import Image
from torchvision import transforms
import pickle
import torch

def load_image(self, filename, transform = None):
    image = Image.open(filename).convert("RGB")
    if transform is not None:
        image = transform(image)
    return image


def save_dataset(filename, target):
    with open("result/dataset/" + filename, "wb") as f:
        pickle.dump(target, f)


def load_dataset(filename):
    with open("result/dataset/" + filename, "rb") as f:
        return pickle.load(f)

def save_model(model, filename):
    torch.save(model.state_dict(), "result/model/" + filename)
