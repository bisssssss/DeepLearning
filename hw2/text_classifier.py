import numpy as np 
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torchtext.vocab as vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BOWclassifier(nn.Module):

    def __init__(self, V, labels=2):
        super(BOWclassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(V, labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class Gembdclassifier(nn.Module):

    def __init__(self, D, labels=2):
        super(Gembdclassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(D, labels),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.fc(x)
        return x

class LSTMclassifier(nn.Module):

    def __init__(self, D, H, labels=2):
        super(LSTMclassifier, self).__init__()

        self.D = D
        self.H = H
        self.p = labels

        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=H,
            batch_first=True
        )

        '''
        self.embd = nn.Embedding(
            num_embeddings=V,
            embedding_dim=D,
            padding_idx=padding_idx
        )
        self.embd.weight.requires_grad=False
        '''
        self.fc = nn.Sequential(
            nn.Linear(H, labels),
            nn.Sigmoid()
        )

    def init_embd(self, weight_matrix):
        self.embd = nn.Embedding(weight_matrix.shape[0], weight_matrix.shape[1])
        self.embd.load_state_dict({'weight': weight_matrix})

    def forward(self, x):
        # x : (N, T, D) already embedded
        x = self.embd(x)
        x, (hn, cn) = self.lstm(x) # (N, T, H), (1, N, H), (1, N, H)
        hn = hn.view(-1, self.H)
        hn = self.fc(hn)
        return hn

class RNNclassifier(nn.Module):

    def __init__(self, D, H, labels=2):
        super(RNNclassifier, self).__init__()

        self.D = D
        self.H = H
        self.p = labels

        self.rnn = nn.RNN(
            input_size=D,
            hidden_size=H,
            batch_first=True
        )
        '''
        self.embd = nn.Embedding(
            num_embeddings=V,
            embedding_dim=D,
            padding_idx=padding_idx
        )
        self.embd.weight.requires_grad=False
        '''
        self.fc = nn.Sequential(
            nn.Linear(H, labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x : (N, T, D) already embedded
        x, hn = self.rnn(x) # (N, T, H), (1, N, H), (1, N, H)
        hn = hn.view(-1, self.H)
        hn = self.fc(hn)
        return hn

class Embdclassifier(nn.Module):

    def __init__(self, V, D, labels=2):
        super(Embdclassifier, self).__init__()
        self.embd = nn.Embedding(V, D)
        #self.embd.weight.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(D, labels),
            nn.Sigmoid()
        )

    def get_xlength(self, xlength):
        self.xlength = xlength

    def forward(self, x):
        # x : (N, T, C)
        x = self.embd(x) # (N, T, D)
        x = torch.sum(x, dim=1) / self.xlength # (N, D)
        x = self.fc(x)
        return x

def create_bow(data, word_to_idx, UNK_IDX=0):

    bow = torch.zeros((len(data), len(word_to_idx))) # (N, V)
    i = 0
    for line in data:
        for word in line:
            try:
                bow[i, word_to_idx[word]] += 1
            except:
                bow[i, UNK_IDX] += 1
        i += 1

    return bow

def create_glove_embd_avg(data, glove, EMBD_DIM):

    embd_data = torch.zeros((len(data), EMBD_DIM))
    i = 0
    for line in data:
        avgpool = torch.zeros(EMBD_DIM, ).type(torch.FloatTensor)
        for word in line:
            try:
                vec = glove.vectors[glove.stoi[word]]
            except:
                # vec = torch.tensor(np.random.normal(scale=0.6, size=(EMBD_DIM, ))).type(torch.FloatTensor)
                vec = torch.zeros((EMBD_DIM, )).type(torch.FloatTensor)
            avgpool += vec
        embd_data[i, :] = avgpool / len(line)
        i += 1

    return  embd_data

def create_pad_glove_embd(data, glove, word_to_idx, EMBD_DIM, maxlen, PAD_IDX=0):

    embd_data = torch.zeros((len(data), maxlen, EMBD_DIM)).type(torch.FloatTensor)
    padvec = glove.vectors[PAD_IDX]
    i = 0
    for line in data:
        j = 0
        for word in line:
            try:
                vec = glove.vectors[word_to_idx[word]]
            except:
                vec = torch.tensor(np.random.normal(scale=0.6, size=(EMBD_DIM, ))).type(torch.FloatTensor)
            embd_data[i, j, :] = vec
            j += 1
            if j == maxlen:
                break
        while j < maxlen:
            embd_data[i, j, :] = padvec
            j += 1
        i += 1

    return  embd_data

def create_glove_weight(glove, word_to_idx, EMBD_DIM, PAD_IDX=0, SPEC_IDX=1):

    embd = torch.zeros((len(word_to_idx), EMBD_DIM)).type(torch.FloatTensor)

    i = 0
    for word in word_to_idx:
        try:
            embd[i,:] = glove.vectors[word_to_idx[word]]
        except:
            embd[i,:] = torch.tensor(np.random.normal(scale=0.6, size=(EMBD_DIM, ))).type(torch.FloatTensor)
        i += 1

    return  embd

def create_pad(data, word_to_idx, maxlen, PAD_IDX=0, UNK_IDX=1):

    dict_data = torch.zeros((len(data), maxlen), dtype=torch.long)
    xlength = torch.zeros((len(data), 1), dtype=torch.int8).type(torch.FloatTensor)
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
        xlength[i,0] = j
        while j < maxlen:
            dict_data[i, j] = PAD_IDX
            j += 1
        i += 1

    return  dict_data, xlength

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

def train(trainloader, valloader, net, criterion, optimizer, device,
          trainlength=None, vallength=None, batchsize=1000, epoch=20):

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

            if trainlength is not None:
                trainlength_batch = trainlength[batch_idx, :].reshape(-1, 1).to(device)
                net.get_xlength(trainlength_batch)
            scores = net(Xtrain_batch)
            running_loss = criterion(scores, labels_batch)

            running_loss.backward()
            optimizer.step()

            # train acc
            pred_label = torch.argmax(scores, dim=1)
            train_acc = torch.sum(pred_label == labels_batch).item() / batchsize

            # val acc
            if valloader is None:
                val_acc = 0.
            else:
                val_batch_idx = np.random.choice(len(Xval), batchsize, replace=False)
                if vallength is not None:
                    vallength_batch = vallength[val_batch_idx,:].reshape(-1, 1).to(device)
                    net.get_xlength(vallength_batch)
                Xval_batch = Xval[val_batch_idx,:].to(device)
                labelsval_batch = labelsval[val_batch_idx].to(device)
                pred_label_val = torch.argmax(net(Xval_batch), dim=1)
                val_acc = torch.sum(pred_label_val == labelsval_batch).item() / batchsize

            #running_loss += loss.item()
            if i % 10 == 0 or i == batchnum-1:
                end = time.time()
                print('[epoch %d, iter %3d] \n     - loss: %.8f     - train acc: %.3f   - val acc : %.3f    - eplased time %.3f' %
                      (epochi + 1, i + 1, running_loss.item() / batchsize, train_acc, val_acc, end-start))
                start = time.time()
    print('Finished Training')

def test(testloader, net, device, testlength=None):

    Xtest = testloader['X'].to(device)
    labelstest = testloader['y'].to(device)
    if testlength is not None:
        testlength = testlength.to(device)
        net.get_xlength(testlength)
    pred_label_test = torch.argmax(net(Xtest), dim=1)
    test_acc = torch.sum(pred_label_test == labelstest).item() / len(Xtest)

    print('\ntest acc: %.3f' %(test_acc))

def task1(data_train, data_val, data_test, data_unlabelled):

    # baseline
    # bag-of-words
    torch.manual_seed(598)
    word_to_idx = {}
    word_to_idx['<UNK>'] = 0
    for line in data_train['X'] + data_val['X']:
        for word in line:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    # nn
    V = len(word_to_idx)
    net = BOWclassifier(V, labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # creating bow dataset
    # train set
    Xtrainbow = create_bow(data_train['X'], word_to_idx).to(device)
    ylabels = torch.tensor(data_train['y'], dtype=torch.long)
    trainset = {'X':Xtrainbow, 'y':ylabels}
    # val set
    Xvalbow = create_bow(data_val['X'], word_to_idx).to(device)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.long)
    valset = {'X': Xvalbow, 'y': ylabelsval}
    # test set
    Xtestbow = create_bow(data_test['X'], word_to_idx).to(device)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.long)
    testset = {'X': Xtestbow, 'y': ylabelstest}

    # training
    print("training BOW model")
    train(trainset, valset, net, criterion, optimizer, device, batchsize=1000, epoch=40)

    # testing
    test(testset, net, device)

    # predicing unlabelled data
    Xunlbow = create_bow(data_unlabelled['X'], word_to_idx).to(device)
    pred_unl = torch.argmax(net(Xunlbow), dim=1)

    f = open("predictions_q1.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def task5(data_train, data_val, data_test, data_unlabelled):

    torch.manual_seed(598)
    EMBD_DIM = 300
    HIDDEN_DIM = 64
    mask = False

    # nn
    net = LSTMclassifier(D=EMBD_DIM, H=HIDDEN_DIM, labels=2).to(device)
    if mask == True:
        criterion = SFMloss
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # creating pretained dataset
    word_to_idx = {}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<SPEC>'] = 1
    datatot = data_train['X'] + data_val['X']
    for line in datatot:
        for word in line:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    glove = vocab.GloVe() # V = 2196017
    maxlen = len(max(datatot, key=len))

    weight_matrix = create_glove_weight(glove, word_to_idx, EMBD_DIM, PAD_IDX=0, SPEC_IDX=1)
    net.init_embd(weight_matrix)
    net = net.to(device)

    # creating dataset
    # train set
    Xtrainrnn,_ = create_pad(data_train['X'], word_to_idx, maxlen)
    #Xtrainrnn = create_pad(data_train['X'], word_to_idx, maxlen)
    ylabels = torch.tensor(data_train['y'], dtype=torch.long)
    trainset = {'X': Xtrainrnn.to(device), 'y': ylabels}
    # val set
    Xvalrnn,_ = create_pad(data_val['X'], word_to_idx, maxlen)
    #Xvalrnn = create_pad(data_val['X'], word_to_idx, maxlen)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.long)
    valset = {'X': Xvalrnn.to(device), 'y': ylabelsval}
    # test set
    Xtestrnn,_ = create_pad(data_test['X'], word_to_idx, maxlen)
    #Xtestrnn = create_pad(data_test['X'], word_to_idx, maxlen)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.long)
    testset = {'X': Xtestrnn.to(device), 'y': ylabelstest}

    # training
    print("training LSTM model")
    train(trainset, valset, net, criterion, optimizer, device, batchsize=1000, epoch=40)

    # testing
    test(testset, net, device)

    # predicing unlabelled data
    Xunlbow,_ = create_pad(data_unlabelled['X'], word_to_idx, maxlen)
    pred_unl = torch.argmax(net(Xunlbow.to(device)), dim=1)

    f = open("predictions_q5.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def task4(data_train, data_val, data_test, data_unlabelled):

    torch.manual_seed(598)
    EMBD_DIM = 300
    HIDDEN_DIM = 64

    # nn
    net = RNNclassifier(D=EMBD_DIM, H=HIDDEN_DIM, labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # creating pretained dataset
    word_to_idx = {}
    word_to_idx['<PAD>'] = 0
    for line in data_train['X'] + data_val['X']:
        for word in line:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    glove = vocab.GloVe() # V = 2196017
    datatot = data_train['X'] + data_val['X'] + data_test['X'] + data_unlabelled['X']
    maxlen = len(max(datatot, key=len))

    # creating dataset
    # train set
    Xtrainrnn = create_pad_glove_embd(data_train['X'], glove, word_to_idx, EMBD_DIM, maxlen).to(device)
    ylabels = torch.tensor(data_train['y'], dtype=torch.long)
    trainset = {'X': Xtrainrnn, 'y': ylabels}
    # val set
    Xvalbow = create_pad_glove_embd(data_val['X'], glove, word_to_idx, EMBD_DIM, maxlen).to(device)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.long)
    valset = {'X': Xvalbow, 'y': ylabelsval}
    # test set
    Xtestbow = create_pad_glove_embd(data_test['X'], glove, word_to_idx, EMBD_DIM, maxlen).to(device)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.long)
    testset = {'X': Xtestbow, 'y': ylabelstest}

    # training
    print("training RNN model")
    train(trainset, valset, net, criterion, optimizer, device, batchsize=1000, epoch=80)

    # testing
    test(testset, net, device)

    # predicing unlabelled data
    Xunlbow = create_pad_glove_embd(data_unlabelled['X'], glove, word_to_idx, EMBD_DIM, maxlen).to(device)
    pred_unl = torch.argmax(net(Xunlbow), dim=1)

    f = open("predictions_q4.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def task2(data_train, data_val, data_test, data_unlabelled):

    torch.manual_seed(598)
    EMBD_DIM = 300

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
    net = Embdclassifier(V, D=EMBD_DIM, labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # creating dataset
    # train set
    Xtrainembd, trainlength = create_pad(data_train['X'], word_to_idx, maxlen, PAD_IDX=0)
    ylabels = torch.tensor(data_train['y'], dtype=torch.long)
    trainset = {'X': Xtrainembd.to(device), 'y': ylabels}

    # val set
    Xvalembd, vallength = create_pad(data_val['X'], word_to_idx, maxlen, PAD_IDX=0)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.long)
    valset = {'X': Xvalembd.to(device), 'y': ylabelsval}
    # test set
    Xtestbow, testlength = create_pad(data_test['X'], word_to_idx, maxlen, PAD_IDX=0)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.long)
    testset = {'X': Xtestbow.to(device), 'y': ylabelstest}

    # training
    print("training EMBD AvgPool model")
    train(trainset, valset, net, criterion, optimizer, device,
          trainlength = trainlength, vallength = vallength, batchsize=1000, epoch=10)

    # testing
    test(testset, net, device, testlength)

    # predicing unlabelled data
    Xunlembd, unllength = create_pad(data_unlabelled['X'], word_to_idx, maxlen, PAD_IDX=0)
    net.get_xlength(unllength.to(device))
    pred_unl = torch.argmax(net(Xunlembd.to(device)), dim=1)

    f = open("predictions_q2.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def SFMloss(x, y, mask):

    N, T, V = x.shape
    xm = x.reshape(N * T, V)
    den = np.exp(xm)
    sftm = den / np.sum(den, axis=1, keepdims=True)
    loss = (1 / N) * np.sum(-np.log(sftm[np.arange(N * T,), y.reshape(N * T,)]) * mask.reshape(N * T,))
    dx = sftm.copy()
    dx[np.arange(N * T), y.reshape(N * T,)] -= 1
    dx *= (1 / N) * mask.reshape(N * T, 1)

    return loss, dx.reshape(N, T, V)

def task3(data_train, data_val, data_test, data_unlabelled):

    torch.manual_seed(598)
    EMBD_DIM = 300 # GloVe default

    #glove = vocab.GloVe(name='42B', dim=EMBD_DIM)
    #glove = vocab.GloVe(name='6B', dim=EMBD_DIM)
    glove = vocab.GloVe()

    # nn
    net = Gembdclassifier(EMBD_DIM, labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # creating dataset
    # train set
    Xtrainrnn = create_glove_embd_avg(data_train['X'], glove, EMBD_DIM).to(device)
    ylabels = torch.tensor(data_train['y'], dtype=torch.long)
    trainset = {'X': Xtrainrnn, 'y': ylabels}

    # val set
    Xvalbow = create_glove_embd_avg(data_val['X'], glove, EMBD_DIM).to(device)
    ylabelsval = torch.tensor(data_val['y'], dtype=torch.long)
    valset = {'X': Xvalbow, 'y': ylabelsval}
    # test set
    Xtestbow = create_glove_embd_avg(data_test['X'], glove, EMBD_DIM).to(device)
    ylabelstest = torch.tensor(data_test['y'], dtype=torch.long)
    testset = {'X': Xtestbow, 'y': ylabelstest}

    # training
    print("training GloVe EMBD AvgPool model")
    train(trainset, valset, net, criterion, optimizer, device, batchsize=1000, epoch=80)

    # testing
    test(testset, net, device)

    # predicing unlabelled data
    Xunlbow = create_glove_embd_avg(data_unlabelled['X'], glove, EMBD_DIM).to(device)
    pred_unl = torch.argmax(net(Xunlbow), dim=1)

    f = open("predictions_q3.txt", "w+")
    for i in range(len(pred_unl)):
        f.write("%d\n" % (pred_unl[i].item()))
    f.close()

def comparision(fname1, fname2):

    f1 = np.loadtxt(fname1)
    f2 = np.loadtxt(fname2)
    print( 1 - sum(f1 == f2) / len(f1))

def cmptot():
    
    for i in range(5):
        for j in range(i+1, 5):
            print(i+1, j+1)
            comparision("predictions_q"+str(i+1)+".txt", "predictions_q"+str(j+1)+".txt")    

def main():

    data_train = read_data('data/train.txt')
    data_val = read_data('data/dev.txt')
    data_test = read_data('data/test.txt')
    data_unlabelled = read_data('data/unlabelled.txt', ylabel=False)

    #task1(data_train, data_val, data_test, data_unlabelled)
    #task2(data_train, data_val, data_test, data_unlabelled)
    #task3(data_train, data_val, data_test, data_unlabelled)
    #task4(data_train, data_val, data_test, data_unlabelled)
    task5(data_train, data_val, data_test, data_unlabelled)


if __name__ == "__main__":
    main()
    #cmptot()
