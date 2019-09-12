from helper import *
from component import *
from config import *

from tqdm import tqdm
from collections import Counter
# from random import shuffle
import random

import nltk
import pickle
import json
import csv

# holds entire token
tokens = { }

# load tokens
with open(FLICKR30k_ANNOTATION, 'r', encoding="utf-8") as f:
    for row in f:
        cols = row.split('\t')
        img = cols[0].split("#")
        caption = cols[1].strip()
        caption = caption.lower()

        if img[0] not in tokens:
            tokens[img[0]] = [ ]

        tokens[img[0]].append(caption.split(' '))

train_set, validation_set, test_set = prase_data_by_ratio(tokens, validation_and_test_dataset_size)

# ###########################################################################################
# STRUCTURE
# ###########################################################################################

train_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

validation_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

# test_dataset =  { "filename" : [ [], [], [], [], [] ] }
test_dataset = {

}

# ###########################################################################################
# TEST
# ###########################################################################################

for filename in test_set:

    # add key
    test_dataset[filename] = []

    # add sentences
    for sentence in test_set[filename]:
        test_dataset[filename].append(sentence)

print("test_set count", len(test_dataset))
save_dataset(dataset_file["test"], test_dataset)


# ###########################################################################################
# VALIDATION
# ###########################################################################################

word_counter = {}
validation_skip_count = 0
filecount = 0
for filename in validation_set:
    filecount += 1

    for sentence in validation_set[filename]:
        item = { "filename" : filename, "caption" : sentence }
        validation_dataset["data"].append(item)

        # increase counter
        for word in sentence:
            word_counter[word] = (word_counter[word] + 1) if word in word_counter else 1

for word in word_counter:
    if word_counter[word] >= term_frequency_threshold:
        validation_dataset["corpus"].append(word)
    else :
        validation_skip_count += 1

random.shuffle(validation_dataset["data"])
save_dataset(dataset_file["validation"], validation_dataset)
print("skip", "validation_skip_count", validation_skip_count)
print("validation image count", filecount)


# ###########################################################################################
# TRAIN
# ###########################################################################################

word_counter = {}
train_skip_count = 0
filecount = 0

for filename in train_set:
    filecount += 1

    for sentence in train_set[filename]:
        item = { "filename" : filename, "caption" : sentence }
        train_dataset["data"].append(item)

        # increase counter
        for word in sentence:
            word_counter[word] = (word_counter[word] + 1) if word in word_counter else 1

for word in word_counter:
    if word_counter[word] >= term_frequency_threshold:
        train_dataset["corpus"].append(word)
    else :
        train_skip_count += 1

save_dataset(dataset_file["train"], train_dataset)
print("skip", "train_skip_count", train_skip_count)
print("trian corpus count", len(train_dataset["corpus"]))
print("train image count", filecount)