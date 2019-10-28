import pycuda.driver as cuda

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from TIMIT import FeatureExtraction, Classifier
from TIMIT import TIMITDataSet
from TIMIT import phone_code
from TIMIT import weights_init


torch.cuda.memory_allocated()  # the amount of GPU memory allocated   
torch.cuda.memory_cached()     # the amount of GPU memory cached   

torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.


#----------------------------------------------------------------------------------


def trainClassifier(classifier, training_loader, learning_rate, momentum, epochCnt=10):
    #
    # choise one of the following optimizers
    #optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)   # create a stochastic gradient descent optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adadelta(classifier.parameters())
    #
    #for opt in optimizer.param_groups: opt['lr']=0.5e-4
    #
    lossFunc = nn.NLLLoss()
    #
    for epoch in range(0, epochCnt):
        idx=0
        for input_batch, target_batch in training_loader:   # to iterate: input_batch, target_batch = next(iter(training_loader))
            #
            input_batch  = input_batch.to('cuda')    # the shape of input_batch = (batch_size, 1, FREQ_BAND, SOUND_WINDOW)
            target_batch = target_batch.to('cuda')   # the shape of target_batch = (batch_size)
            #
            optimizer.zero_grad()
            #
            lnSoftmax = nn.LogSoftmax(dim=1)
            #
            output_batch = classifier(input_batch)
            loss = lossFunc(lnSoftmax(output_batch), target_batch)
            loss.backward()
            # run gradient descent based on the gradients calculated from the backward() function
            optimizer.step()    
            #
            if (idx+1)%5==0 or idx==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, idx * len(input_batch), len(training_loader.dataset),
                               100. * idx / len(training_loader), loss.data))
            idx += 1
        #
    #


def testClassifier(classifier, testing_loader):
    #
    bIdx = 0
    totalCorrect = 0
    totalRuns = 0
    #
    for input_batch, target_batch in testing_loader:   # to iterate: input_batch, target_batch = next(iter(testing_loader))
        #
        bIdx += 1
        input_batch  = input_batch.to('cuda')    # the shape of input_batch = (batch_size, 1, FREQ_BAND, SOUND_WINDOW)
        target_batch = target_batch.to('cuda')   # the shape of target_batch = (batch_size)
        #        
        output_batch = classifier(input_batch)
        sm = F.softmax(output_batch, dim=1)
        #
        numcorrect = 0
        numruns = np.shape(target_batch)[0]
        #
        for _ in range (0, numruns):
            v, i = torch.max(sm[_], 0)
            if i == target_batch[_]: 
                numcorrect += 1
                totalCorrect += 1
            #
            totalRuns += 1
        #
        #print('Batch #',  bIdx, "; Number of correctly identified phonemes ", numcorrect)    
        #
    percentCorrect = totalCorrect*100/totalRuns
    print('Percent identified correctly: ' , percentCorrect, '%')
    return percentCorrect
    #



#----------------------------------------------------------------------------------
# train a classifier

TIMITRootPath = "C:\\Data\\TIMIT\\"
TIMITDataPath = "C:\\Data\\TIMIT\\Data\\"
#dialects = ['DR1']
dialects = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']

trainingDataFrame = "train_data.df"
testingDataFrame  = "test_data.df"

training_data_set = TIMITDataSet(TIMITRootPath, dialects, trainingDataFrame, phonedict, dataCnt=140000)
training_loader = DataLoader(training_data_set, batch_size=200, shuffle=True, num_workers=0)

testing_data_set = TIMITDataSet(TIMITRootPath, dialects, testingDataFrame, phonedict)
testing_loader = DataLoader(testing_data_set, batch_size=100, shuffle=True, num_workers=0)

classifier = Classifier().cuda()
classifier.apply(weights_init)


runNum = 1
epochCnt=1
parameters = []
progress = []


learning_rate = 1e-4
momentum = 0.0
parameters.append([runNum, learning_rate, momentum])

#
for r in range(0, 30):
    for i in range (0, 10):
        print ('Training epoch #: ', i)
        trainClassifier(classifier, training_loader, learning_rate, momentum, epochCnt)
        #progress.append([runNum, testClassifier(classifier, testing_loader)])
        runNum += 1
    #
    progress.append([runNum, testClassifier(classifier, testing_loader)])
#
    
    
plt.plot(progress)

accuList=[]
for [epoch, accu] in progress:
    accuList.append(accu)
plt.plot(accuList)


np.savetxt('progress_batch50.txt', progress)
np.savetxt('parameters_batch50.txt', parameters)

progress = np.loadtxt('progress_batch130.txt').tolist()
parameters = np.loadtxt('parameters_batch130.txt').tolist()



'''
frmCnt=training_data_set.frmCnt
totlFrmCnt=0
for _ in range(0, len(frmCnt)):
    totlFrmCnt += frmCnt[_]
'''

'''
# load the classifier
classifierPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
classifierName = 'Classifier_Inception_TIMIT_ALL.pt'
classifierModelPathName = classifierPath + classifierName
# load the trained classifier neural network
classifier0 = Classifier().cuda()
classifier0.load_state_dict(torch.load(classifierModelPathName))
classifier0.eval()

testClassifier(classifier0, testing_loader)
'''


#----------------------------------------------------------------------------------
# save the trained classifier model
    
modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'Classifier_TIMIT_ALL_new_batch130.pt'
modelName = 'Classifier_TIMIT_ALL_new_batch50.pt'
#modelName = 'Classifier_Inception_TIMIT_ALL.pt'
modelPathName = modelPath + modelName

torch.save(classifier.state_dict(), modelPathName)

classifier.parameterCnt()


# load a trained classifier 
classifier = Classifier().cuda()
classifier.load_state_dict(torch.load(modelPathName))
classifier.eval()



optimizer=None
classifier=None
torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.
torch.cuda.memory_cached()     # the amount of GPU memory cached   


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

    
def trainAudioGenerator(generator, training_data_set, classifierModelPathName, batch_size=100, epochCnt=50):
    #
    training_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    learning_rate = 1e-6
    momentum = 0.0
    #
    optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    #
    # to change the optimizer learning rate
    for opt in optimizer.param_groups: opt['lr']=1e-7
    #
    #lossFunc = nn.L1Loss()
    lossFunc = nn.MSELoss()
    #
    for epoch in range(0, epochCnt):
        batchCnt=0
        totalCnt=0
        #
        for inputs, labels in training_loader:   # to iterate: inputs, labels = next(iter(training_loader))
            #
            cnt=0
            #inputs = inputs * (inputs>1e-3).float()
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            #
            # pass the data through the classifier first
            outputs = classifier(inputs)
            sm = F.softmax(outputs, dim=1)
            #
            # loop through the result and pick only the correctly classified ones for training the generator
            candidates = []
            for i in range(0, len(inputs)): 
                _, idx = torch.max(sm[i], 0)
                if labels[i]==idx:
                    candidates.append(True)
                else:
                    candidates.append(False)
            #
            cnt = sum(1  for x in candidates if x)
            totalCnt += cnt
            #
            if cnt>0:
                optimizer.zero_grad()
                #
                features = classifier.featureMatrix
                #features = features * (features>0).float()
                audios = generator(features[candidates])
                loss = lossFunc(audios, inputs[candidates][:, 0:1, :, 5:6])
                loss.backward()
                # run gradient descent based on the gradients calculated from the backward() function
                optimizer.step()    
            #
            if (batchCnt+1)%5==0 or batchCnt==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, totalCnt, len(training_loader.dataset), loss.data))
            #
            batchCnt += 1
        #
    #


#----------------------------------------------------------------------------------
# train an audio generator
    
classifierPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
classifierName = 'Classifier_Inception_TIMIT_DR1.pt'
classifierName = 'Classifier_Inception_TIMIT_ALL.pt'
classifierName = 'Classifier_TIMIT_DR1.pt'
classifierModelPathName = classifierPath + classifierName


# load the trained classifier neural network
classifier = Classifier().cuda()
classifier.load_state_dict(torch.load(classifierModelPathName))
classifier.eval()
    

generator = AudioGenerator().cuda()
generator.apply(weights_init)

optimizer=None
generator=None
torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.
torch.cuda.memory_cached()     # the amount of GPU memory cached   

#----------------------------------------------------------------------------------
# save the trained audio generator

modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'Generator_Inception_TIMIT.pt'
modelPathName = modelPath + modelName

torch.save(generator.state_dict(), modelPathName)


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


def trainSoftMask(generator, training_data_set, classifierModelPathName, batch_size=100, epochCnt=100):
    #
    training_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    learning_rate = 1e-3
    momentum = 0.0
    #
    optimizer = torch.optim.SGD(softmask.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(softmask.parameters(), lr=learning_rate)
    #
    # to change the optimizer learning rate
    for opt in optimizer.param_groups: opt['lr']=1e-3
    #
    #lossFunc = nn.L1Loss()
    lossFunc = nn.MSELoss()
    #
    for epoch in range(0, epochCnt):
        batchCnt=0
        totalCnt=0
        #
        for [inMatrix, original, noisy], classId in training_loader:   # to iterate: [inMatrix, original, noisy], classId = next(iter(training_loader))
             #
            totalCnt += len(original)
            #
            inMatrix = inMatrix.to('cuda')
            #
            diff = noisy - original
            diff = diff.to('cuda')
            #
            #weight = original/noisy
            #flag = weight>2.0
            #weight[flag]=2.0
            #weight = weight.to('cuda')
            #
            # pass the data through the classifier first
            outputs = classifier(inMatrix)
            featureMap = classifier.featureMatrix
            #
            optimizer.zero_grad()
            mask = softmask(featureMap)
            #loss = lossFunc(mask, weight)
            loss = lossFunc(mask, diff)
            loss.backward()
            # run gradient descent based on the gradients calculated from the backward() function
            optimizer.step()    
            #
            if (batchCnt+1)%5==0 or batchCnt==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, totalCnt, len(training_loader.dataset), loss.data))
            #
            batchCnt += 1
        #
    #
    
def trainRecursiveSoftMask(generator, training_data_set, classifierModelPathName, batch_size=100, epochCnt=400):
    #
    training_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    learning_rate = 1e-3
    momentum = 0.0
    #
    optimizer = torch.optim.SGD(softmask.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(softmask.parameters(), lr=learning_rate)
    #
    # to change the optimizer learning rate
    for opt in optimizer.param_groups: opt['lr']=1e-6
    #
    #lossFunc = nn.L1Loss()
    lossFunc = nn.MSELoss()
    #
    for epoch in range(0, epochCnt):
        batchCnt=0
        totalCnt=0
        #
        for [inMatrix, original, noisy], classId in training_loader:   # to iterate: [inMatrix, original, noisy], classId = next(iter(training_loader))
             #
            totalCnt += len(original)
            #
            inMatrix = inMatrix.to('cuda')
            #
            diff = noisy - original
            diff = diff.to('cuda')
            #
            #weight = original/noisy
            #flag = weight>2.0
            #weight[flag]=2.0
            #weight = weight.to('cuda')
            #
            # pass the data through the classifier first
            outputs = classifier(inMatrix)
            featureMap = classifier.featureMatrix
            #
            optimizer.zero_grad()
            mask = softmask(featureMap)
            #loss = lossFunc(mask, weight)
            loss = lossFunc(mask, diff)
            loss.backward()
            # run gradient descent based on the gradients calculated from the backward() function
            optimizer.step()    
            #
            if (batchCnt+1)%5==0 or batchCnt==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, totalCnt, len(training_loader.dataset), loss.data))
            #
            batchCnt += 1
        #
    #


#----------------------------------------------------------------------------------
# train an soft mask generator
    
classifierPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
classifierName = 'Classifier_Inception_TIMIT_DR1.pt'
classifierName = 'Classifier_Inception_TIMIT_ALL.pt'
classifierName = 'Classifier_TIMIT_DR1.pt'
classifierModelPathName = classifierPath + classifierName


# load the trained classifier neural network
classifier = Classifier().cuda()
classifier.load_state_dict(torch.load(classifierModelPathName))
classifier.eval()


TIMITRootPath = "C:\\Data\\TIMIT\\"
TIMITDataPath = "C:\\Data\\TIMIT\\Data\\"
dialects = ['DR2']
dialects = ['DR3', 'DR4']
dialects = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']

training_data_set = TIMITNoisyDataSet(TIMITRootPath, dialects, phonedict)
training_loader = DataLoader(training_data_set, batch_size=200, shuffle=True, num_workers=0)

softmask = AudioGenerator().cuda()
softmask.apply(weights_init)

optimizer=None
softmask=None
torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.
torch.cuda.memory_cached()     # the amount of GPU memory cached   

for i in range(0, 100):
    print(i, weight[i].max())
    
#----------------------------------------------------------------------------------
# save the trained softmask

modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'SoftMask_Subtraction_Inception_TIMIT_ALL.pt'
modelPathName = modelPath + modelName

torch.save(softmask.state_dict(), modelPathName)

#----------------------------------------------------------------------------------
# load a trained softmask


modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'SoftMask_Inception_TIMIT.pt'
modelPathName = modelPath + modelName

softmask = AudioGenerator().cuda()
softmask.load_state_dict(torch.load(modelPathName))
softmask.eval()
