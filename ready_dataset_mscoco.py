from helper import *
from component import *
from config import *

from tqdm import tqdm
from collections import Counter
from random import shuffle

import nltk
import pickle
import json
import csv

# csv setting
csv.register_dialect('quote_dialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)

# complete_dataset
complete_dataset = {  }   # format: { "filename" : [] }

for path in [COCO_TRAIN_ANNOTATION, COCO_VALIDATION_ANNOTATION]:

    with open(path) as f:
        coco_annotation = json.load(f)

    coco_image = { }
    for item in coco_annotation["images"]:
        coco_image[item["id"]] = item

    for anno_item in tqdm(coco_annotation["annotations"]):

        # retrieve item information
        caption = nltk.tokenize.word_tokenize(anno_item["caption"].lower())
        filename = coco_image[anno_item["image_id"]]["file_name"]

        # check if already added
        if filename not in complete_dataset:
            complete_dataset[filename] = [ ]

        complete_dataset[filename].append(caption)

train_set, validation_set, test_set = prase_data_by_ratio(complete_dataset, validation_and_test_dataset_size)

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


# ###########################################################################################
# TEST
# ###########################################################################################

save_dataset(dataset_file["test"], test_set)


# ###########################################################################################
# VALIDATION
# ###########################################################################################

validation_skip_count = 0
word_counter = {}
for filename in tqdm(validation_set):
    for sentence in validation_set[filename]:

        # increase counter
        for word in sentence:
            word_counter[word] = (word_counter[word] + 1) if word in word_counter else 1

        # add item to data
        validation_dataset["data"].append({ "filename" : filename, "caption" : sentence })

with open(RESULT_DATASET_PATH + dataset_skip_file['validation'], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='quote_dialect')
    for word in tqdm(word_counter):
        if word_counter[word] >= term_frequency_threshold:
            validation_dataset["corpus"].append(word)
        else:
            validation_skip_count += 1
            spamwriter.writerow([word, word_counter[word]])

shuffle(validation_dataset["data"])

save_dataset(dataset_file["validation"], validation_dataset)

print("skip", "validation_skip_count", validation_skip_count)


# ###########################################################################################
# TRAIN
# ###########################################################################################

train_skip_count = 0
word_counter = {}
for filename in tqdm(train_set):
    for sentence in train_set[filename]:

        # increase counter
        for word in sentence:
            word_counter[word] = (word_counter[word] + 1) if word in word_counter else 1

        # add item to data
        train_dataset["data"].append({ "filename" : filename, "caption" : sentence })

with open(RESULT_DATASET_PATH + dataset_skip_file['train'], 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='quote_dialect')
    for word in word_counter:
        if word_counter[word] >= term_frequency_threshold:
            train_dataset["corpus"].append(word)
        else:
            train_skip_count += 1
            spamwriter.writerow([word, word_counter[word]])

shuffle(train_dataset["data"])

save_dataset(dataset_file["train"], train_dataset)

print("skip", "train_skip_count", train_skip_count)


# ###########################################################################################
# PRETRAIN
# ###########################################################################################

word_counter = {"noun":{},"pronoun":{},"verb":{},"adjective":{},"adverb":{},"conjunction":{},"preposition":{},"interjection":{}}
pos_skip_counter = {"noun":0,"pronoun":0,"verb":0,"adjective":0,"adverb":0,"conjunction":0,"preposition":0,"interjection":0}

for filename in tqdm(train_set):
    for sentence in train_set[filename]:
        tokens_pos = nltk.pos_tag(sentence)
        for word,tag in tokens_pos:
            for nltkpos in nltk_to_pos:
                if tag in nltk_to_pos[nltkpos]:

                    # word count
                    word_counter[nltkpos][word] = (word_counter[nltkpos][word] + 1) if word in word_counter[nltkpos] else 1

                    # increase
                    pretrain_dataset[nltkpos]["data"].append({ "filename" : filename, "word" : word })

for pos in word_counter:
    with open(RESULT_DATASET_PATH + dataset_skip_file[pos], 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='quote_dialect')
        for word in word_counter[pos]:
            if word_counter[pos][word] >= term_frequency_threshold:
                pretrain_dataset[pos]["corpus"].append(word)
            else:
                pos_skip_counter[pos] += 1
                spamwriter.writerow([word, word_counter[pos][word]])

for pos in pos_skip_counter:
    print("skip", pos, pos_skip_counter[pos])

save_dataset(dataset_file["pretrain"], pretrain_dataset)




#
