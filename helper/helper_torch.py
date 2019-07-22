import torch

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
