import pickle
import torch
import os

from config import *
from component import *
from helper import *
from model import *
from PIL import Image

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

import torch.nn as nn
from torchvision import transforms

from score import run_score
from tqdm import tqdm

class Pcr():

    def __init__(self, train_dataset):

        # ############################################################
        # Variable LOADING
        # ############################################################

        # GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # image preprocessing
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # input size
        # output size of CNN and input size of RNN
        self.embed_size = 256

        # load encoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(train_dataset["corpus"])


        # ############################################################
        # Object LOADING
        # ############################################################

        # load CNN models
        self.noun_model         = self.load_cnn_model("noun")
        self.verb_model         = self.load_cnn_model("verb")
        self.adjective_model    = self.load_cnn_model("adjective")
        self.conjunction_model  = self.load_cnn_model("conjunction")
        self.preposition_model  = self.load_cnn_model("preposition")

        # load RNN model
        self.rnn_model = self.load_rnn_model( len(train_dataset["corpus"]) )


        # ############################################################
        # GPU LOADING
        # ############################################################

        # send all models to GPU
        self.noun_model = self.noun_model.to(self.device, non_blocking = True)
        self.verb_model = self.verb_model.to(self.device, non_blocking = True)
        self.adjective_model = self.adjective_model.to(self.device, non_blocking = True)
        self.conjunction_model = self.conjunction_model.to(self.device, non_blocking = True)
        self.preposition_model = self.preposition_model.to(self.device, non_blocking = True)
        self.rnn_model = self.rnn_model.to(self.device, non_blocking = True)
        self.rnn_model.module = self.rnn_model.module.to(self.device, non_blocking = True)


    def load_cnn_model(self, pos):
        features = torch.load(model_save_path + model_file[pos]["train"])
        cnn_model = Encoder(embed_size = self.embed_size)
        cnn_model = nn.DataParallel(cnn_model)
        cnn_model.to(self.device, non_blocking=True)
        cnn_model.load_state_dict(features)
        cnn_model.eval() # change model mode to eval
        return cnn_model


    def load_rnn_model(self, train_output_size):
        features = torch.load(model_save_path + model_file["decoder"]["train"])
        decoder_model = Decoder( (self.embed_size * 5), decoder_hidden_size, train_output_size, lstm_number_of_layers)
        decoder_model = nn.DataParallel(decoder_model, device_ids = [0])
        decoder_model.to(self.device, non_blocking=True)
        decoder_model.load_state_dict(features)
        decoder_model.eval() # change model mode to eval
        return decoder_model


    def load_image_file(self, filename_with_full_path):
        image  = Image.open(filename_with_full_path)
        image = image.resize([224, 224], Image.LANCZOS)
        image = self.transform(image).unsqueeze(0)
        return image


    def testcase(self, image_file_full_path):

        # load image
        image = self.load_image_file(image_file_full_path)

        # send all objects to GPU
        image = image.to(self.device, non_blocking = True)

        # encoder image
        noun_features           = self.noun_model(image)
        verb_features           = self.verb_model(image)
        adjective_features      = self.adjective_model(image)
        conjunction_features    = self.conjunction_model(image)
        preposition_features    = self.preposition_model(image)

        # combine encoder features
        features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)

        # send features to GPU
        features = features.to(self.device, non_blocking = True)

        # decoder
        sampled_ids = self.rnn_model.module.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # # reverse label encoding
        sampled_caption = []
        for word_id in sampled_ids:
            inverted_label = self.label_encoder.inverse_transform([word_id])
            word = inverted_label[0]
            sampled_caption.append(word)
            if word == '<end>':
                break

        return sampled_caption[1:-1]


if __name__ == "__main__":

    # train dataset
    train_dataset = load_dataset(dataset_file["train"])

    # test dataset
    test_dataset = load_dataset(dataset_file["test"])

    # create PCR
    pcr = Pcr(train_dataset)

    # result holder
    result_holder = []

    # skip count
    skip_count = 0

    # create image yes or no
    should_create_image = True

    # the loading bar
    pbar = tqdm(test_dataset)

    # create hypothesis
    for filename in pbar:
        try:
            hypothesis = pcr.testcase(coco_test_image_path + filename)
            result_holder.append({
                "reference" : test_dataset[filename]["caption"],
                "hypothesis" : hypothesis
            })
            pbar.set_description( filename + " " + ' '.join(hypothesis) )

            # create image with file and caption
            if should_create_image:
                lst = ["Reference"]
                lst.extend([' '.join(cap) for cap in test_dataset[filename]["caption"]])
                lst.extend(["", "Hypothesis"])
                lst.append(' '.join(hypothesis))
                create_image_caption(coco_test_image_path_original + filename, image_with_caption + filename, lst)
        except:
            skip_count += 1

     # save result to file
    with open(os.path.join(base_path, "result", dataset_file["result"]), "wb") as f:
        pickle.dump(result_holder, f)

    print("skip_count:", skip_count)

    # run scoring
    run_score(result_holder)
