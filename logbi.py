import pickle
import os

from config import *
from component import *
from helper import *
from model import *

from tqdm import tqdm

db_type = 3

dataset_file = {
    "pretrain"          : "cnn_pretrain_dataset.pkl",
    "train"             : "cnn_rnn_train_dataset.pkl",
    "test"              : "cnn_rnn_test_dataset.pkl",
    "validation"        : "cnn_rnn_validation_dataset.pkl",
    "result"            : "cnn_rnn_caption_result.pkl"
}

outputfile = "mscoco"

if db_type == 2:
    dataset_file["pretrain"] =      "cnn_pretrain_dataset_flickr8k.pkl"
    dataset_file["train"] =         "cnn_rnn_train_dataset_flickr8k.pkl"
    dataset_file["test"] =          "cnn_rnn_test_dataset_flickr8k.pkl"
    dataset_file["validation"] =   "cnn_rnn_validation_dataset_flickr8k.pkl"
    dataset_file["result"] =        "cnn_rnn_caption_result_flickr8k.pkl"
    outputfile = "flickr8k"
elif db_type == 3:
    dataset_file["pretrain"] = "cnn_pretrain_dataset_flickr30k.pkl"
    dataset_file["train"] = "cnn_rnn_train_dataset_flickr30k.pkl"
    dataset_file["test"] = "cnn_rnn_test_dataset_flickr30k.pkl"
    dataset_file["validation"] = "cnn_rnn_validation_dataset_flickr30k.pkl"
    dataset_file["result"] = "cnn_rnn_caption_result_flickr30k.pkl"
    outputfile = "flickr30k"

train_dataset = load_dataset(RESULT_DATASET_PATH + dataset_file["train"])
test_dataset = load_dataset(RESULT_DATASET_PATH + dataset_file["test"])
validation_dataset = load_dataset(RESULT_DATASET_PATH + dataset_file["validation"])

with open(RESULT_ROOT + "/"+outputfile+"_train.txt", "w") as f:
    for item in tqdm(train_dataset["data"]):
        f.write(item["filename"]+"\n")

with open(RESULT_ROOT + "/"+outputfile+"_test.txt", "w") as f:
    for filename in tqdm(test_dataset):
        f.write(filename+"\n")

with open(RESULT_ROOT + "/"+outputfile+"_validation.txt", "w") as f:
    for item in tqdm(validation_dataset["data"]):
        f.write(item["filename"]+"\n")

























#
