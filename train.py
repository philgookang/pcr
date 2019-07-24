from model import *
from component import *
from helper import *
from config import *

import pickle
import torch
import numpy as np
import torch.nn as nn
import visdom
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

def train(learning_rate, use_visdom):

    # #################################################################################
    # VARIABLES
    # #################################################################################

    # pretrain dataset
    pretrain_dataset = load_dataset(dataset_file["pretrain"])

    # train dataset
    train_dataset = load_dataset(dataset_file["train"])

    # size of each training cycle
    batch_size = 128

    # number of epochs
    number_epochs = 6

    # number of process worker
    number_of_workers = 32

    # dimension of lstm hidden states
    hidden_size = 512

    # number of layers in lstm
    number_of_layers = 1

    # number of GPU devices
    numer_of_devices = torch.cuda.device_count()

    # randomize dataset
    is_shuffle = True

    # image random crop size
    crop_size = 224

    # log step size
    log_step = 5

    # output size of CNN and input size of RNN
    embed_size = 256

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print device
    print("device: ", device)

    trans =  transforms.Compose([
        transforms.RandomCrop(image_crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # create visdom graph
    vis = visdom.Visdom()

    # graph pointer
    loss_graph = None

    # #################################################################################
    # DATASET INITIALIZE
    # #################################################################################

    loader = torch.utils.data.DataLoader(dataset=TrainDataset(dataset=train_dataset,transform=trans),
                                         batch_size = train_batch_size,
                                         shuffle = is_shuffle,
                                         num_workers = number_of_workers,
                                         collate_fn = coco_collate_fn)

    # Build the model
    noun_model = load_model("noun", len(pretrain_dataset["noun"]["corpus"]), device, embed_size)
    verb_model = load_model("verb", len(pretrain_dataset["verb"]["corpus"]), device, embed_size)
    adjective_model = load_model("adjective", len(pretrain_dataset["adjective"]["corpus"]), device, embed_size)
    conjunction_model = load_model("conjunction", len(pretrain_dataset["conjunction"]["corpus"]), device, embed_size)
    preposition_model = load_model("preposition", len(pretrain_dataset["preposition"]["corpus"]), device, embed_size)
    decoder_model = Decoder( (embed_size * 5) , hidden_size, len(train_dataset["corpus"]), number_of_layers)

    # send to GPU
    decoder_model = decoder_model.to(device, non_blocking = True)

    # parallelize model
    decoder_model = nn.DataParallel(decoder_model, device_ids = [0])

    # loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # parameters
    params = list(decoder_model.parameters()) + list(preposition_model.module.linear.parameters()) + list(preposition_model.module.bn.parameters()) + list(conjunction_model.module.linear.parameters()) + list(conjunction_model.module.bn.parameters()) + list(adjective_model.module.linear.parameters()) + list(adjective_model.module.bn.parameters()) + list(verb_model.module.linear.parameters()) + list(verb_model.module.bn.parameters()) + list(noun_model.module.linear.parameters()) + list(noun_model.module.bn.parameters())

    # optimization
    optimizer = torch.optim.Adam(params, lr = learning_rate)


    # #################################################################################
    # TRAINING
    # #################################################################################

    noun_model.cuda(device)
    verb_model.cuda(device)
    adjective_model.cuda(device)
    conjunction_model.cuda(device)
    preposition_model.cuda(device)

    # train the model
    total_step = len(loader)
    for epoch in range(number_epochs):
        for i, (images, captions, lengths) in enumerate(loader):

            # set tensor to GPU
            images = images.to(device, non_blocking = True)
            captions = captions.to(device, non_blocking = True)

            # create minibatch
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # forward propagation
            noun_features = noun_model(images)
            verb_features = verb_model(images)
            adjective_features = adjective_model(images)
            conjunction_features = conjunction_model(images)
            preposition_features = preposition_model(images)
            features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)
            outputs = decoder_model(features, captions, lengths)

            # backpropagation
            loss = criterion(outputs, targets)
            noun_model.zero_grad()
            verb_model.zero_grad()
            adjective_model.zero_grad()
            conjunction_model.zero_grad()
            preposition_model.zero_grad()
            decoder_model.zero_grad()
            loss.backward()
            optimizer.step()

            # eval
            loss_val = loss.item()

            # check for every step
            if i % log_step == 0:

                # graph loss and process
                if use_visdom:
                    if loss_graph == None:
                        legend = ["Epoch {0}".format(epoch) for epoch in range(number_epochs)]
                        loss_graph = vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), opts=dict(
                                                  title = "(NVACP)CNN-RNN (LR: {0})".format(learning_rate), legend=legend, xlabel='Iteration', ylabel='Loss'
                                              ))
                    else:
                        vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), update='append', win=loss_graph)

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, number_epochs, i+1, total_step, loss_val))

    # return trained model
    return noun_model, verb_model, adjective_model, conjunction_model, preposition_model, decoder_model


def load_model(pos, embed_size, device, new_embed_size):
    features = torch.load("result/model/" + model_file[pos]["pretrain"])
    cnn_model = Encoder(embed_size = embed_size)
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to(device, non_blocking=True)
    cnn_model.load_state_dict(features)
    cnn_model.module.update_layer(new_embed_size)
    return cnn_model


if __name__ == "__main__":

    # learning rate
    learning_rate = [ 0.004 ]

    # back stats
    cudnn.benchmark = True

    # loop through each different learning rate
    for lr in learning_rate:

        # train PoS CNN - RNN
        noun, verb, adjective, conjunction, preposition, decoder = train(lr, True)

        # save trained model
        save_model(noun, model_file["noun"]["train"])
        save_model(verb, model_file["verb"]["train"])
        save_model(adjective, model_file["adjective"]["train"])
        save_model(conjunction, model_file["conjunction"]["train"])
        save_model(preposition, model_file["preposition"]["train"])
        save_model(decoder, model_file["decoder"]["train"])
