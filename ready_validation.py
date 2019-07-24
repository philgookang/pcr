from helper import *
from component import *
from config import *

from tqdm import tqdm
from random import shuffle

import nltk
import pickle
import json

# coco initalize
with open(coco_caption_path_validation) as f:
    train_coco = json.load(f)

# coco image holder
image_holder = { }
for item in train_coco["images"]:
    image_holder[item["id"]] = item

# ###################################################
print("load complete dataset")
# ###################################################

# complete_dataset
validation_set = { }

# this list holds { "filename" : {"id", "caption"} }
for item in tqdm(train_coco["annotations"]):

    # retrieve item information
    caption = str(item["caption"])
    caption = nltk.tokenize.word_tokenize(caption.lower())
    filename = image_holder[item["image_id"]]["file_name"]

    # check if already added
    if filename not in validation_set:
        validation_set[filename] = { "reference" : [] }

    validation_set[filename]["reference"].append(caption)


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
