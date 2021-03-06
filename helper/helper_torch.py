import torch
from config import *

def coco_collate_fn(data):
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def combine_output(noun_features, verb_features, adjective_features, conjunction_features, preposition_features, device, decoder_model):
    features = ((noun_features + verb_features + adjective_features + conjunction_features + preposition_features) / 5)
    return features, None
