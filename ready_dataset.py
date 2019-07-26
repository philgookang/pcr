from helper import *
from component import *
from config import *

from tqdm import tqdm
from collections import Counter
from random import shuffle

import nltk
import pickle
import json

# coco initalize
with open(coco_caption_path_train) as f:
    train_coco = json.load(f)

with open(coco_caption_path_validation) as f:
    validation_coco = json.load(f)

# coco image holder
image_holder = { }
for item in train_coco["images"]:
    image_holder[item["id"]] = item

validation_image_holder = { }
for item in validation_coco["images"]:
    validation_image_holder[item["id"]] = item

# ###################################################
print("load complete dataset")
# ###################################################

# complete_dataset
train_set = { }
validation_set = { }

# this list holds { "filename" : {"id", "caption"} }
for item in tqdm(train_coco["annotations"]):

    # retrieve item information
    caption = str(item["caption"])
    filename = image_holder[item["image_id"]]["file_name"]

    # check if already added
    if filename not in train_set:
        train_set[filename] = { "id" : [], "caption" : [], "image_id" : [] }

    train_set[filename]["id"].append(id)
    train_set[filename]["caption"].append(caption)

# this list holds { "filename" : {"id", "caption"} }
for item in tqdm(validation_coco["annotations"]):

    # retrieve item information
    caption = str(item["caption"])
    caption = nltk.tokenize.word_tokenize(caption.lower())
    filename = validation_image_holder[item["image_id"]]["file_name"]

    # check if already added
    if filename not in validation_set:
        validation_set[filename] = { "reference" : [] }

    validation_set[filename]["reference"].append(caption)

# ###################################################
print("pretrain & train dataset")
# ###################################################

pretrain_dataset = {
    "noun" : { "corpus" : [], "data" : [] },
    "pronoun" : { "corpus" : [], "data" : [] },
    "verb" : { "corpus" : [], "data" : [] },
    "adjective" : { "corpus" : [], "data" : [] },
    "adverb" : { "corpus" : [], "data" : [] },
    "conjunction" : { "corpus" : [], "data" : [] },
    "preposition" : { "corpus" : [], "data" : [] },
    "interjection": { "corpus" : [], "data" : [] }
}

train_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

counter = Counter()

for filename in tqdm(train_set):
    item = train_set[filename]

    for id, caption in tqdm(zip(item["id"], item["caption"])):

        # break caption string up
        tokens = nltk.tokenize.word_tokenize(caption.lower())

        # -- Logic of pretrain CNN model
        tokens_pos = nltk.pos_tag(tokens)
        for word,tag in tqdm(tokens_pos):
            for nltkpos in nltk_to_pos:
                if tag in nltk_to_pos[nltkpos]:
                    pretrain_dataset[nltkpos]["data"].append({ "filename"  : filename, "word" : word })
                    if word not in pretrain_dataset[nltkpos]["corpus"]:
                        pretrain_dataset[nltkpos]["corpus"].append(word)

        # -- Logic of CNN-RNN training
        counter.update(tokens)
        train_dataset["data"].append({ "filename" : filename, "caption" : tokens })

# loop through and check threshold
words = [word for word, cnt in counter.items() if cnt >= threshold]

# loop through words that pass threshold
for word in words: train_dataset["corpus"].append(word)

# save dataset to file
save_dataset(dataset_file["pretrain"], pretrain_dataset)
save_dataset(dataset_file["train"], train_dataset)


# ###################################################
print("validation dataset")
# ###################################################

validation_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

for filename in tqdm(validation_set):
    for word_list in validation_set[filename]["reference"]:
        for word in word_list:
            if word not in validation_dataset["corpus"]:
                validation_dataset["corpus"].append(word)

        validation_dataset["data"].append({ "filename" : filename, "caption" : word_list })

shuffle(validation_dataset["data"])

save_dataset(dataset_file["validation"], validation_dataset)

#######

print("PoS", "Corpus", "Data")
for k in pretrain_dataset:
    print(k, len(pretrain_dataset[k]["corpus"]), len(pretrain_dataset[k]["data"]))
