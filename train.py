from model import *
from component import *
from helper import *
from config import *

import matplotlib.pyplot as plt

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

    # train dataset
    validation_dataset = load_dataset(dataset_file["validation"])

    # number of process worker
    number_of_workers = 32

    # randomize dataset
    is_shuffle = True

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
    vis = visdom.Visdom() if use_visdom else None

    # graph pointer
    loss_graph = None

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience = 7, verbose = True)

    # to track the training loss as the model trains
    train_losses = []

    # to track the validation loss as the model trains
    valid_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # #################################################################################
    # DATASET INITIALIZE
    # #################################################################################

    # create train loader
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=TrainEvalDataset(dataset=train_dataset, transform=trans, image_path=coco_train_image_path),
        batch_size=train_batch_size, shuffle=is_shuffle, num_workers=number_of_workers, collate_fn=coco_collate_fn)
    eval_dataset_loader = torch.utils.data.DataLoader(
        dataset=TrainEvalDataset(dataset=validation_dataset, transform=trans, image_path=coco_validation_image_path),
        batch_size=train_batch_size, shuffle=is_shuffle, num_workers=number_of_workers, collate_fn=coco_collate_fn)

    # Build Encoder
    noun_model = load_model("noun", len(pretrain_dataset["noun"]["corpus"]), device, embed_size)
    verb_model = load_model("verb", len(pretrain_dataset["verb"]["corpus"]), device, embed_size)
    adjective_model = load_model("adjective", len(pretrain_dataset["adjective"]["corpus"]), device, embed_size)
    conjunction_model = load_model("conjunction", len(pretrain_dataset["conjunction"]["corpus"]), device, embed_size)
    preposition_model = load_model("preposition", len(pretrain_dataset["preposition"]["corpus"]), device, embed_size)

    # Build Decoer
    decoder_model = Decoder( (embed_size * 5) , decoder_hidden_size, len(train_dataset["corpus"]), lstm_number_of_layers)
    decoder_model = decoder_model.to(device, non_blocking = True) # send to GPU
    decoder_model = nn.DataParallel(decoder_model, device_ids = [0]) # parallelize model

    # loss function & optimization
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    params = list(decoder_model.parameters()) + list(preposition_model.module.linear.parameters()) + list(preposition_model.module.bn.parameters()) + list(conjunction_model.module.linear.parameters()) + list(conjunction_model.module.bn.parameters()) + list(adjective_model.module.linear.parameters()) + list(adjective_model.module.bn.parameters()) + list(verb_model.module.linear.parameters()) + list(verb_model.module.bn.parameters()) + list(noun_model.module.linear.parameters()) + list(noun_model.module.bn.parameters())
    optimizer = torch.optim.Adam(params, lr = learning_rate)

    # #################################################################################
    # TRAINING
    # #################################################################################

    # send all to GPU
    noun_model.cuda(device)
    verb_model.cuda(device)
    adjective_model.cuda(device)
    conjunction_model.cuda(device)
    preposition_model.cuda(device)

    # train & eval stats
    total_step = len(train_dataset_loader)

    # train & eval the model
    for epoch in range(cnn_rnn_number_epochs):

        # prep model for training
        noun_model.train()
        verb_model.train()
        adjective_model.train()
        conjunction_model.train()
        preposition_model.train()
        decoder_model.train()
        for i, (images, captions, lengths) in enumerate(train_dataset_loader):

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

            # record training loss
            train_losses.append(loss_val)

            # check for every step
            if i % log_step == 0:
                # graph loss and process
                if use_visdom: loss_graph = log_graph(loss_graph, loss_val, cnn_rnn_number_epochs, epoch, i, vis)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, cnn_rnn_number_epochs, i + 1, total_step, loss_val))
            break

        # prep model for validation
        noun_model.eval()
        verb_model.eval()
        adjective_model.eval()
        conjunction_model.eval()
        preposition_model.eval()
        decoder_model.eval()
        for i, (images, captions, lengths) in enumerate(eval_dataset_loader):
            noun_features = noun_model(images)
            verb_features = verb_model(images)
            adjective_features = adjective_model(images)
            conjunction_features = conjunction_model(images)
            preposition_features = preposition_model(images)
            features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)
            features = features.to(device, non_blocking=True)
            outputs = decoder_model(features, captions, lengths)
            loss = criterion(outputs, targets)

            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print('Result [{}/{}], Train {:.5f}, Valid: {:.5f}'.format(epoch + 1, cnn_rnn_number_epochs, train_loss, valid_loss))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, decoder_model)

        if early_stopping.early_stop:
            print("Early stopping!!!!!!!!!!!!!!!!!!!!!!")
            print("Early stopping!!!!!!!!!!!!!!!!!!!!!!")
            print("Early stopping!!!!!!!!!!!!!!!!!!!!!!")
            print("Early stopping!!!!!!!!!!!!!!!!!!!!!!")
            break

        break

    # graph losses
    graph_early_stop(train_loss, valid_loss)

    # return trained model
    return noun_model, verb_model, adjective_model, conjunction_model, preposition_model, decoder_model


def log_graph(loss_graph, loss_val, number_epochs, epoch, i, vis):
    if loss_graph == None:
        legend = ["Epoch {0}".format(epoch) for epoch in range(number_epochs)]
        loss_graph = vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), opts=dict(
            title="(NVACP)CNN-RNN (LR: {0})".format(learning_rate), legend=legend, xlabel='Iteration', ylabel='Loss'
        ))
    else:
        vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), update='append', win=loss_graph)
    return loss_graph


def load_model(pos, embed_size, device, new_embed_size):
    features = torch.load("result/model/" + model_file[pos]["pretrain"])
    cnn_model = Encoder(embed_size = embed_size)
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to(device, non_blocking=True)
    cnn_model.load_state_dict(features)
    cnn_model.module.update_layer(new_embed_size)
    return cnn_model


def graph_early_stop(train_loss, valid_loss):
    # (visdom.Visdom()).line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), opts=dict(
    #     title="Early Stop Result",
    #     legend=["Training Loss", "Validation Loss"],
    #     xlabel='Epoch',
    #     ylabel='Loss'
    # ))

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


if __name__ == "__main__":

    # learning rate
    learning_rate = [ 0.004 ]

    # back stats
    cudnn.benchmark = True

    # loop through each different learning rate
    for lr in learning_rate:

        # train PoS CNN - RNN
        noun, verb, adjective, conjunction, preposition, decoder = train(lr, False)

        # save trained model
        save_model(noun, model_file["noun"]["train"])
        save_model(verb, model_file["verb"]["train"])
        save_model(adjective, model_file["adjective"]["train"])
        save_model(conjunction, model_file["conjunction"]["train"])
        save_model(preposition, model_file["preposition"]["train"])
        save_model(decoder, model_file["decoder"]["train"])
