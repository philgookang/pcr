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
from torchvision import transforms

def train(pos, dataset, learning_rate, use_visdom):

    # #################################################################################
    # VARIABLES
    # #################################################################################

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    # target size
    embed_size = len(dataset["corpus"])
    print("embed_size: ", embed_size)

    trans =  transforms.Compose([transforms.RandomCrop(image_crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # create visdom graph
    vis = visdom.Visdom()
    loss_graph = None


    # #################################################################################
    # DATASET INITIALIZE
    # #################################################################################

    # data loader
    loader = torch.utils.data.DataLoader(dataset=PretrainDataset(dataset=dataset,transform=trans),
                                         batch_size = pretrain_batch_size,
                                         shuffle = is_shuffle,
                                         num_workers = number_of_workers,
                                         collate_fn = coco_collate_fn)

    # model to train
    cnn_model = Encoder(embed_size = embed_size)
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to(device, non_blocking = True)

    # loss function & optimization
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = learning_rate)


    # #################################################################################
    # TRAINING
    # #################################################################################

    # train the model
    total_step = len(loader)
    for epoch in range(pretrain_number_epochs):
        for i, (images, labels, lengths) in enumerate(loader):

            # set tensor to GPU
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            # forward propagation
            outputs = cnn_model(images)

            # backpropagation
            loss = criterion(outputs,  torch.max(labels, 1)[1] )
            cnn_model.zero_grad()
            loss.backward()
            optimizer.step()

            # eval
            loss_val = loss.item()

            # check for every step
            if i % 36 == 0:
                if use_visdom:
                    if loss_graph == None:
                        legend = ["Epoch {0}".format(epoch) for epoch in range(pretrain_number_epochs)]
                        loss_graph = vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), opts=dict(
                                                  title = "{0} (LR: {1})".format(pos, learning_rate), legend=legend, xlabel='Iteration', ylabel='Loss'
                                              ))
                    else:
                        vis.line(X=np.array([i]), Y=np.array([loss_val]), name="Epoch {0}".format(epoch), update='append', win=loss_graph)

                print(pos, 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, pretrain_number_epochs, i+1, total_step, loss_val))

    # return trained model
    return cnn_model

if __name__ == "__main__":

    # get dataset
    dataset = load_dataset(dataset_file["pretrain"])

    # learning rate
    learning_rate = [ 0.004 ] # 0.006, 0.005, , 0.003, 0.004

    # back stats
    cudnn.benchmark = True

    # loop through list
    for pos in ["adjective", "verb", "conjunction", "preposition"]: # noun", "

        # loop through each different learning rate
        for lr in learning_rate:

            # train PoS CNN
            trained_model = train(pos, dataset[pos], lr, True)

            # save trained model
            save_model(trained_model, model_file[pos]["pretrain"])
