import torch
import pickle
import os
from PIL import Image, PSDraw, ImageDraw, ImageFont
from config import *

def save_dataset(filename, target):
    with open(os.path.join(base_path, "result", "dataset", filename), "wb") as f:
        pickle.dump(target, f)

def load_dataset(filename):
    with open(os.path.join(base_path, "result", "dataset", filename), "rb") as f:
        return pickle.load(f)

def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(base_path, "result", "model", filename))

def create_image_caption(original, target, lst):

    font = os.path.join(base_path, "config", "RobotoRegular.ttf")

    img = Image.open(original, 'r')
    w, h = img.size
    img = img.crop((0,0,w + 900,h))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, 20)
#     text_w, text_h = draw.textsize(lst[0]) #, font
    for no,txt in enumerate(lst):
        draw.text((w + 10, 2 + (37*no)), txt, (255,255,255), font=font)
    img.save(target)
