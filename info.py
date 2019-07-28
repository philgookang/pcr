from model import *
from component import *
from helper import *
from config import *

print("####################################################################################")
print("#","Pretrain Dataset")
print("####################################################################################")

# pretrain dataset
pretrain_dataset = load_dataset(dataset_file["pretrain"])
for key in nltk_to_pos:
    print(key, "corpus:", len(pretrain_dataset[key]["corpus"]), "data:", len(pretrain_dataset[key]["data"]))

print("####################################################################################")
print("#","Train, Validation, Test Dataset")
print("####################################################################################")

# train dataset
train_dataset = load_dataset(dataset_file["train"])
print("Train", "corpus:", len(train_dataset["corpus"]), "data:", len(train_dataset["data"]))

# validation dataset
validation_dataset = load_dataset(dataset_file["validation"])
print("Validation", "corpus:", len(validation_dataset["corpus"]), "data:", len(validation_dataset["data"]))

# test dataset
test_dataset = load_dataset(dataset_file["test"])
print("Test", "data:", len(test_dataset))
