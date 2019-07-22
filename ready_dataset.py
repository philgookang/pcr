from helper import *
from component import *
from config import *

from pycocotools.coco import COCO
from tqdm import tqdm
from collections import Counter

import nltk
import pickle

# coco initalize
coco = COCO(coco_caption_path)
ids = coco.anns.keys()

# ###################################################
print("load complete dataset")
# ###################################################

# complete_dataset
complete_dataset = { }

# this list holds { "filename" : {"id", "caption"} }
for i, id in tqdm(enumerate(ids)):

    # retrieve item information
    caption = str(coco.anns[id]["caption"])
    filename = coco.loadImgs(coco.anns[id]["image_id"])[0]["file_name"]

    # check if already added
    if filename not in complete_dataset:
        complete_dataset[filename] = { "id" : [], "caption" : [], "image_id" : [] }

    complete_dataset[filename]["id"].append(id)
    complete_dataset[filename]["caption"].append(caption)
    complete_dataset[filename]["image_id"].append(coco.anns[id]["image_id"])


# ###################################################
print("parse dataset by train, validation, test")
# ###################################################

train_set, validation_set, test_set = prase_data_by_ratio(complete_dataset, 5000)

save_dataset(dataset_file["train_org"], train_set)
save_dataset(dataset_file["validation_org"], validation_set)
save_dataset(dataset_file["test_org"], test_set)


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

save_dataset(dataset_file["pretrain"], pretrain_dataset)
save_dataset(dataset_file["train"], train_dataset)


# ###################################################
print("validation & test dataset")
# ###################################################

validation_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

test_dataset = {
    "corpus" : ["<pad>", "<start>", "<end>", "<unk>"], # complete list of all the words used
    "data" : [] # { "filename" : "", "caption" : "" }
}

for filename in tqdm(validation_set):
    item = validation_set[filename]
    for id, caption in tqdm(zip(item["id"], item["caption"])):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        validation_dataset["data"].append({ "filename" : filename, "caption" : tokens })
        for word in tokens:
            if word not in validation_dataset["corpus"]:
                validation_dataset["corpus"].append(word)

for filename in tqdm(test_set):
    item = test_set[filename]
    for id, caption in tqdm(zip(item["id"], item["caption"])):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        test_dataset["data"].append({ "filename" : filename, "caption" : tokens })
        for word in tokens:
            if word not in test_dataset["corpus"]:
                test_dataset["corpus"].append(word)

save_dataset(dataset_file["validation"], validation_dataset)
save_dataset(dataset_file["test"], test_dataset)
