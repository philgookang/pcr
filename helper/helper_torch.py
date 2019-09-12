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

def coco_collate_fn_for_bidirectional(data):
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

    x = []
    y = []
    l = []
    l2 = []

    for image_id, sentence in enumerate(targets):

        targets = torch.zeros(len(sentence)).long()

        # section = [ ]

        for index in range(len(sentence)-1):

            current = sentence[index]
            next = sentence[(index+1)]

            targets[index] = current

            # section.append( targets.clone() )
            x.append( { "section" : image_id, "data" : targets.clone(), "target" : int(next.cpu().detach().numpy()) } )
            y.append( int(next.cpu().detach().numpy()) )
            l2.append(1)

    def sort_by_zero_count(ten):
        cnt = 0
        for i in ten["data"]:
            if i != 0:
                cnt += 1
        return cnt

    x.sort(key=sort_by_zero_count, reverse=True)

    lengthlength = []

    for item in x:
        lengthlength.append([item["target"]])
        l.append((sort_by_zero_count(item) + 1))

    yyy = torch.FloatTensor(lengthlength).long()

    return images, x, yyy, l, l2

def combine_vertically(*args):
    lst = []
    for a, b, c, d, e in zip(*args):
        z = torch.cat((a, b, c, d, e))
        lst.append(z.cpu().detach().numpy())
    final = torch.tensor(lst)
    return final


def combine_output(noun_features, verb_features, adjective_features, conjunction_features, preposition_features, device, decoder_model):
    features = None
    attributes = None
    if cnn_output_combine_methods == 1:
        features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)
    elif cnn_output_combine_methods == 2:
        features = ((noun_features + verb_features + adjective_features + conjunction_features + preposition_features) / 5)
        # features = ((noun_features + verb_features + adjective_features + conjunction_features) / 4) # preposition_features
    elif cnn_output_combine_methods == 3:
        features = noun_features
        attributes = ((verb_features + adjective_features + conjunction_features + preposition_features) / 4)
        attributes = attributes.to(device, non_blocking = True)
    elif cnn_output_combine_methods == 4:
        features = torch.cat((noun_features, verb_features, adjective_features, conjunction_features, preposition_features), 1)
        features = decoder_model.module.linear_combiner(features)
    elif cnn_output_combine_methods == 5:
        features = torch.cat((noun_features, verb_features, adjective_features, conjunction_features, preposition_features), 1)
        features = decoder_model.module.linear_combiner(features)
        features = decoder_model.module.dropout(features)
        features = decoder_model.module.linear_combiner2(features)
        features = decoder_model.module.dropout2(features)
    features = features.to(device, non_blocking = True)
    return features, attributes


def flip_tensor(lst, is_grad = False):
    '''this only works for 3차원 dimension'''
    if is_grad:
        lst = lst.cpu().detach().numpy()
    else:
        lst = lst.cpu().numpy()

    result_list = [ ]
    for inner_list in lst:
        result_inner_list = []
        for item_list in inner_list:
            result_inner_list.append( list(reversed(item_list)) )
        result_list.append(result_inner_list)
    bb = torch.tensor(result_list)
    return bb
