import torch
import pickle
import os
from config import *

def save_dataset(filename, target):
    with open(os.path.join(base_path, "result", "dataset", filename), "wb") as f:
        pickle.dump(target, f)

def load_dataset(filename):
    with open(os.path.join(base_path, "result", "dataset", filename), "rb") as f:
        return pickle.load(f)

def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(base_path, "result", "model", filename))
