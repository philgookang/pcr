from model import *
from component import *
from config import *
from helper import *

import pickle
import torch
import numpy as np
import torch.nn as nn
import visdom
import torch.backends.cudnn as cudnn

def train(dataset, lr):

    # #################################################################################
    # VARIABLES
    # #################################################################################

    # learning rate
    learning_rate = lr

    # size of each training cycle
    batch_size = 124

    # number of epochs
    number_epochs = 6

    # number of process worker
    number_of_workers = 32

    # randomize dataset
    is_shuffle = True

    # image random crop size
    crop_size = 224

    # log step size
    log_step = 36

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print device
    print("device: ", device)

    # target size
    embed_size = len(dataset["corpus"])

    # print embed size
    print("embed_size: ", embed_size)

    # create visdom graph
    vis = visdom.Visdom()

    # graph pointer
    loss_graph = None

    # #################################################################################
    # DATASET INITIALIZE
    # #################################################################################

    # data loader
    loader = torch.utils.data.DataLoader(dataset=PretrainDataset(dataset=dataset,transform=resnet_transform()),
                                         batch_size = batch_size,
                                         shuffle = is_shuffle,
                                         num_workers = number_of_workers,
                                         collate_fn = coco_collate_fn)

    # model to train
    cnn_model = Encoder(embed_size = embed_size)

    # use multiple devices
    cnn_model = nn.DataParallel(cnn_model)

    # send model to device
    cnn_model.to(device, non_blocking = True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # optimization
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = learning_rate)


    # #################################################################################
    # TRAINING
    # #################################################################################

    # train the model
    total_step = len(loader)
    for epoch in range(number_epochs):
        for i, (images, labels, lengths) in enumerate(loader):

            # set tensor to GPU
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            # reset gradient
            optimizer.zero_grad()

            # forward propagation
            outputs = cnn_model(images)

            # backpropagation
            loss = criterion(outputs,  torch.max(labels, 1)[1] )
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
                                                  title = "{0} (LR: {1})".format(book.target.capitalize(), learning_rate), legend=legend, xlabel='Iteration', ylabel='Loss'
                                              ))
                    else:
                        vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), update='append', win=loss_graph)

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, number_epochs, i+1, total_step, loss_val))

            break
        break

    # return trained model
    return cnn_model

if __name__ == "__main__":

    # get dataset
    dataset = load_dataset(dataset_file["pretrain"])

    # learning rate
    learning_rate = [ 0.004 ] # 0.006, 0.005, , 0.003

    # back stats
    cudnn.benchmark = True

    # loop through list
    for pos in ["adjective"]: #"noun", "verb",  , "conjunction", "preposition"

        # loop through each different learning rate
        for lr in learning_rate:

            # train PoS CNN
            trained_model = train(dataset[pos], lr)

            # save trained model
            save_model(trained_model, model_file[pos]["pretrain"])