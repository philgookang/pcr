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


# fix random seed
random.seed(10)

# holds entire token
tokens = { }

# load tokens
with open(FLICKR8k_ANNOTATION, 'r') as f:
    for row in f:
        cols = row.split('\t')
        img = cols[0].split("#")
        caption = cols[1].strip()
        caption = caption.lower()

        if img[0] not in tokens:
            tokens[img[0]] = [ ]

        tokens[img[0]].append(caption.split(' '))

# ###########################################################################################
# STRUCTURE
# ###########################################################################################

pretrain_dataset = {
    "noun" : { "corpus" : ["<unk>"], "data" : [] },
    "pronoun" : { "corpus" : ["<unk>"], "data" : [] },
    "verb" : { "corpus" : ["<unk>"], "data" : [] },
    "adjective" : { "corpus" : ["<unk>"], "data" : [] },
    "adverb" : { "corpus" : ["<unk>"], "data" : [] },
    "conjunction" : { "corpus" : ["<unk>"], "data" : [] },
    "preposition" : { "corpus" : ["<unk>"], "data" : [] },
    "interjection": { "corpus" : ["<unk>"], "data" : [] }
}

train_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

validation_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

# test_set =  { "filename" : [ [], [], [], [], [] ] }
test_set = {

}

# ###########################################################################################
# TEST
# ###########################################################################################

with open(FLICKR8k_TEST_IMG, 'r') as f:
    for filename in f:
        filename = filename.strip()

        if filename not in test_set:
            test_set[filename] = []

        for sentence in tokens[filename]:
            test_set[filename].append(sentence)

print("test_set count", len(test_set))
save_dataset(dataset_file["test"], test_set)


# ###########################################################################################
# VALIDATION
# ###########################################################################################

word_counter = {}
validation_skip_count = 0
filecount = 0
with open(FLICKR8k_VALIDATION_IMG, 'r') as f:
    for filename in f:
        filename = filename.strip()
        filecount += 1

        for sentence in tokens[filename]:
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
with open(FLICKR8k_TRAIN_IMG, 'r') as f:
    for filename in f:
        filename = filename.strip()
        filecount += 1

        for sentence in tokens[filename]:
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

random.shuffle(train_dataset["data"])
save_dataset(dataset_file["train"], train_dataset)
print("skip", "train_skip_count", train_skip_count)
print("trian corpus count", len(train_dataset["corpus"]))
print("train image count", filecount)



# ###########################################################################################
# PRETRAIN
# ###########################################################################################

word_counter = {"noun":{},"pronoun":{},"verb":{},"adjective":{},"adverb":{},"conjunction":{},"preposition":{},"interjection":{}}
pos_skip_counter = {"noun":0,"pronoun":0,"verb":0,"adjective":0,"adverb":0,"conjunction":0,"preposition":0,"interjection":0}

for item in tqdm(train_dataset["data"]):
    tokens_pos = nltk.pos_tag(item['caption'])
    for word,tag in tokens_pos:
        for nltkpos in nltk_to_pos:
            if tag in nltk_to_pos[nltkpos]:

                # word count
                word_counter[nltkpos][word] = (word_counter[nltkpos][word] + 1) if word in word_counter[nltkpos] else 1

                # increase
                pretrain_dataset[nltkpos]["data"].append({ "filename" : filename, "word" : word })

for pos in word_counter:
    for word in word_counter[pos]:
        if word_counter[pos][word] >= term_frequency_threshold:
            pretrain_dataset[pos]["corpus"].append(word)
        else:
            pos_skip_counter[pos] += 1

for pos in pos_skip_counter:
    print("skip", pos, pos_skip_counter[pos])

save_dataset(dataset_file["pretrain"], pretrain_dataset)
