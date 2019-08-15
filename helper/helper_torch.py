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

def combine_vertically(*args):
    lst = []
    for a, b, c, d, e in zip(*args):
        z = torch.cat((a, b, c, d, e))
        lst.append(z.cpu().detach().numpy())
    final = torch.tensor(lst)
    return final


def combine_output(noun_features, verb_features, adjective_features, conjunction_features, preposition_features, device):
    features = None
    attributes = None
    if cnn_output_combine_methods == 1:
        features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)
    elif cnn_output_combine_methods == 2:
        features = ((noun_features + verb_features + adjective_features + conjunction_features + preposition_features) / 5)
    elif cnn_output_combine_methods == 3:
        features = noun_features
        attributes = ((verb_features + adjective_features + conjunction_features + preposition_features) / 4)
        attributes = attributes.to(device, non_blocking = True)
    features = features.to(device, non_blocking = True)
    return features, attributes
