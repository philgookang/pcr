import torch
import pickle
import os
from PIL import Image, PSDraw, ImageDraw, ImageFont
from config import *

def save_dataset(filename, target):
    with open(os.path.join(RESULT_DATASET_PATH, filename), "wb") as f:
        pickle.dump(target, f)

def load_dataset(filename):
    with open(os.path.join(RESULT_DATASET_PATH, filename), "rb") as f:
        return pickle.load(f)

def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(RESULT_MODEL_PATH, filename))

def create_image_caption(original, target, lst):
    font = os.path.join(BASE_PATH, "rss", "RobotoRegular.ttf")
    img = Image.open(original, 'r')
    w, h = img.size
    img = img.crop((0,0,w + 900,h))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, 20)
    for no,txt in enumerate(lst):
        draw.text((w + 10, 2 + (37*no)), txt, (255,255,255), font=font)
    img.save(target)
