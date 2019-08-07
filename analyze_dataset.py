from component import *
from config import *
from helper import *

import pickle
import torch
import os
import nltk
import json
import csv

from tqdm import tqdm

verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
verb_type = {"VB" : [], "VBD" : [], "VBG" : [], "VBN" : [], "VBP" : [], "VBZ" : []}
verb_type_counter = { }

with open(coco_caption_path_train) as f:
    train_coco = json.load(f)

cnt = 0

for item in tqdm(train_coco["annotations"]):
    cnt += 1

    tokens = nltk.tokenize.word_tokenize(item["caption"].lower())
    tokens_pos = nltk.pos_tag(tokens)

    for (word, pos) in tokens_pos:

        if pos in verbs:

            if word not in verb_type[pos]:
                verb_type[pos].append(word)

            counter = pos + "_" + word
            if counter not in verb_type_counter:
                verb_type_counter[counter] = 0
            verb_type_counter[counter] += 1



csv.register_dialect('quote_dialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)
with open('result/verb_pos.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='quote_dialect')


    spamwriter.writerow(verbs)

    i = 0
    while True:

        VB  = verb_type["VB"][i] +   if i < len(verb_type["VB"]) else ""
        VBD = verb_type["VBD"][i] if i < len(verb_type["VBD"]) else ""
        VBG = verb_type["VBG"][i] if i < len(verb_type["VBG"]) else ""
        VBN = verb_type["VBN"][i] if i < len(verb_type["VBN"]) else ""
        VBP = verb_type["VBP"][i] if i < len(verb_type["VBP"]) else ""
        VBZ = verb_type["VBZ"][i] if i < len(verb_type["VBZ"]) else ""

        if VB == "" and VBD == "" and VBG == "" and VBN == "" and VBP == "" and VBZ == "":
            break

        print(i, VB, VBD, VBG, VBN, VBP, VBZ)

        spamwriter.writerow([VB, VBD, VBG, VBN, VBP, VBZ])
        i += 1
