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
complete_dataset = {  }   # format: { "filename" : {"id" : [], "caption" : []} }

for path in [coco_caption_path_train, coco_caption_path_validation]:

    with open(path) as f:
        coco_annotation = json.load(f)

    coco_image = { }
    for item in coco_annotation["images"]:
        coco_image[item["id"]] = item

    for anno_item in tqdm(train_coco["annotations"]):

        # retrieve item information
        caption = nltk.tokenize.word_tokenize(anno_item["caption"].lower())
        filename = coco_image[anno_item["image_id"]]["file_name"]

        # check if already added
        if filename not in train_set:
            complete_dataset[filename] = { "caption" : [], "image_id" : [] }

        complete_dataset[filename]["caption"].append(caption)
