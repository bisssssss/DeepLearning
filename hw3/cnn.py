import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext.vocab as vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

import math
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):

    def __init__(self, V, F, D=300, kernel=5, mode='avg'):
        super(CNN, self).__init__()
        self.F = F
        self.C = D
        self.embd = nn.Embedding(V, D)
        self.conv = nn.Conv1d(self.C, self.F, kernel_size=kernel, stride=1)
        self.mode = mode
        self.actf = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(F, 1),
            nn.Sigmoid()
        )

    def pool(self, x):
        if self.mode == 'max':
            xpool = torch.max(x, dim=2)[0]
        else:
            xpool = torch.mean(x, dim=2)
        return xpool

    def forward(self, x):
        '''
        INPUT:  (N, H)  2D tensor of word idx
        OUTPUT: (N)     logits
        '''
        x = self.embd(x)
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(-1, self.F)
        x = self.actf(x)
        x = self.fc(x)
        return x

def read_data(fname, ylabel=True):

    with open(fname, 'r') as f:
        data = [line.split() for line in f]

    #random.shuffle(data)

    if ylabel == False:
        return {'X': data}

    X, y = [], []
    for i in range(len(data)):
        y.append(int(data[i][0]))
        X.append(data[i][1:])

    dataset = {'X': X, 'y':y}

    return dataset

def create_pad(data, word_to_idx, PAD_IDX=0, UNK_IDX=1):

    maxlen = len(max(data, key=len))
    dict_data = torch.zeros((len(data), maxlen), dtype=torch.long)
    i = 0
    for line in data:
        j = 0
        for word in line:
            try:
                dict_data[i, j] = word_to_idx[word]
            except:
                dict_data[i, j] = UNK_IDX
            j += 1
            if j == maxlen:
                break
        while j < maxlen:
            dict_data[i, j] = PAD_IDX
            j += 1
        i += 1

    return  dict_data

def train(trainloader, valloader, net, criterion, optimizer, batchsize=1000, epoch=20):

    Xtrain = trainloader['X']
    labels = trainloader['y']
    if valloader != None:
        Xval = valloader['X']
        labelsval = valloader['y']

    for epochi in range(epoch):

        start = time.time()
        running_loss = 0.0

        batchnum = len(Xtrain) // batchsize

        for i in range(batchnum):

            batch_idx = np.random.choice(len(Xtrain), batchsize, replace=False)
            Xtrain_batch = Xtrain[batch_idx,:].to(device)
            labels_batch = labels[batch_idx].to(device)

            optimizer.zero_grad()

            scores = net(Xtrain_batch)
            running_loss = criterion(scores, labels_batch.float())

            running_loss.backward()
            optimizer.step()

            # train acc
            pred_label = (scores>0.5).type(dtype=torch.int8)
            train_acc = torch.sum(pred_label == labels_batch).item() / batchsize

            # val acc
            if valloader is None:
                val_acc = 0.
            else:
                val_batch_idx = np.random.choice(len(Xval), batchsize, replace=False)
                Xval_batch = Xval[val_batch_idx, :].to(device)
                labelsval_batch = labelsval[val_batch_idx].to(device)
                pred_label_val = (net(Xval_batch) > 0.5).type(dtype=torch.int8)
                val_acc = torch.sum(pred_label_val == labelsval_batch).item() / batchsize

            #running_loss += loss.item()
            if i % 10 == 0 or i == batchnum-1:
                end = time.time()
                print('[epoch %d, iter %3d] \n     - loss: %.8f     - train acc: %.3f   - val acc : %.3f    - eplased time %.3f' %
                      (epochi + 1, i + 1, running_loss.item() / batchsize, train_acc, val_acc, end-start))
                start = time.time()
    print('Finished Training')

def test(testloader, net):

    Xtest = testloader['X'].to(device)
    labelstest = testloader['y'].to(device)
    pred_label_test = (net(Xtest) > 0.5).type(dtype=torch.int8)
    test_acc = torch.sum(pred_label_test == labelstest).item() / len(Xtest)

    print('\ntest acc: %.3f' %(test_acc))

def task_cnn(kernel, mode):

    data_train = read_data('../../hw2/data/train.txt')
    data_val = read_data('../../hw2/data/dev.txt')
    data_test = read_data('../../hw2/data/test.txt')
    data_unlabelled = read_data('../../hw2/data/unlabelled.txt', ylabel=False)

    torch.manual_seed(598)
    EMBD_DIM = 300
    F = 128

    # creating pretained dataset
    word_to_idx = {}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    datatot = data_train['X'] + data_val['X']
    for line in datatot:
        for word in line:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    V = len(word_to_idx)
    maxlen = len(max(datatot, key=len))

    # nn
    net = CNN(V, F=F, D=EMBD_DIM, kernel=kernel, mode=mode).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # creating dataset
    # train set
    Xtrainembd= create_pad(data_train['X'], word_to_idx, PAD_IDX=0)
    ylabels = torch.tensor(data_train['y'], dtype=torch.int8).view(-1,1)
    trainset = {'X': Xtrainembd.to(device), 'y': ylabels}

    # val set
    Xvalembd = create_pad(data_val['X'], word_to_idx, PAD_IDX=0)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.int8).view(-1,1)
    valset = {'X': Xvalembd.to(device), 'y': ylabelsval}
    # test set
    Xtestbow = create_pad(data_test['X'], word_to_idx, PAD_IDX=0)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.int8).view(-1,1)
    testset = {'X': Xtestbow.to(device), 'y': ylabelstest}

    # training
    print("training...  model: %spool, kernel=%d " %(mode, kernel))
    train(trainset, valset, net, criterion, optimizer, batchsize=1000, epoch=10)

    # testing
    test(testset, net)

    # predicing unlabelled data
    Xunlembd = create_pad(data_unlabelled['X'], word_to_idx, PAD_IDX=0)
    pred_unl = (net(Xunlembd.to(device)) > 0.5).type(dtype=torch.int8)

    f = open("predictions_cnn.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def main():
    task_cnn(kernel=7, mode='avg')

if __name__ == "__main__":
    main()