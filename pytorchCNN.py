
import pycuda.driver as cuda


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


# the following functions test GPU availability    
torch.cuda.is_available()
cuda.init()

cuda.Device.count()                # number of GPU(s)
id = torch.cuda.current_device()   # Get Id of default device
cuda.Device(id).name()             # Get the name for the device "Id"

torch.cuda.memory_allocated()  # the amount of GPU memory allocated   
torch.cuda.memory_cached()     # the amount of GPU memory cached   

torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.


#----------------------------------------------------------------------------------
SOUND_WINDOW = 300
FREQ_BAND = 400
INPUT_SHAPE = (1, FREQ_BAND, SOUND_WINDOW)

CLASS_COUNT = 10


#----------------------------------------------------------------------------------


class FeatureExtraction(nn.Module):
    """
    This is a sequential model starting with two inception layers at the front, 
    followed by nine convolutional layers of. 
    The output of the last layer is supposed to contain sufficient features to facilitate classification.
    The output of the last layer can be used as the input to a transposed CNN network (deconvolutional network) to
    reconstruct the input.
    """

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        #
        self.inc11 = nn.Conv2d(  1,  10, kernel_size=(45, 1), stride=1, padding=(22, 0))
        self.inc12 = nn.Conv2d(  1,  10, kernel_size=(1, 45), stride=1, padding=(0, 22))
        self.inc13 = nn.Conv2d(  1,  14, kernel_size=(15, 3), stride=1, padding=( 7, 1))
        self.inc14 = nn.Conv2d(  1,  14, kernel_size=( 5, 5), stride=1, padding=( 2, 2))
        #
        self.inc21 = nn.Conv2d( 48,  10, kernel_size=(25,  1), stride=1, padding=(12,  0))
        self.inc22 = nn.Conv2d( 48,  10, kernel_size=( 1, 25), stride=1, padding=( 0, 12))
        self.inc23 = nn.Conv2d( 48,  40, kernel_size=( 5,  5), stride=1, padding=( 2,  2))
        #
        self.conv1 = nn.Conv2d( 60, 100, kernel_size=( 4, 4), stride=2, padding=(1,1))
        #
        self.conv2 = nn.Conv2d(100, 100, kernel_size=( 3 ,3), stride=1)
        #
        self.conv3 = nn.Conv2d(100, 100, kernel_size=( 3, 3), stride=1)
        #
        self.conv4 = nn.Conv2d(100, 100, kernel_size=( 3, 3), stride=1)
        #
        self.conv5 = nn.Conv2d(100, 120, kernel_size=( 2, 2), stride=2)
        #
        self.conv6 = nn.Conv2d(120, 120, kernel_size=( 3, 3), stride=1)
        #
        self.conv7 = nn.Conv2d(120, 140, kernel_size=( 2, 2), stride=(1, 2))
        #
        self.conv8 = nn.Conv2d(140, 160, kernel_size=( 2, 2), stride=(2, 1))
        #
        self.conv9 = nn.Conv2d(160, 200, kernel_size=( 2, 2), stride=(1, 2))


    def forward(self, x):
        #
        x11 = F.leaky_relu(self.inc11(x))
        x12 = F.leaky_relu(self.inc12(x))
        x13 = F.leaky_relu(self.inc13(x))
        x14 = F.leaky_relu(self.inc14(x))
        x1  = torch.cat((x11, x12, x13, x14), dim=1)  # np.shape(x1) = [_, 48, 400, 300]
        #
        x21 = F.leaky_relu(self.inc21(x1))
        x22 = F.leaky_relu(self.inc22(x1))        
        x23 = F.leaky_relu(self.inc23(x1))
        x2  = torch.cat((x21, x22, x23), dim=1)  # np.shape(x2) = [_, 60, 400, 300]
        #
        c1 = F.leaky_relu(self.conv1(x2))  # np.shape(c1) = [_, 100, 200, 150]]
        c2 = F.leaky_relu(self.conv2(c1))  # np.shape(c2) = [_, 100, 198, 148]]
        c3 = F.leaky_relu(self.conv3(c2))  # np.shape(c3) = [_, 100, 196, 146]]
        c4 = F.leaky_relu(self.conv4(c3))  # np.shape(c4) = [_, 100, 194, 144]]
        c5 = F.leaky_relu(self.conv5(c4))  # np.shape(c5) = [_, 100,  97,  72]]
        c6 = F.leaky_relu(self.conv6(c5))  # np.shape(c6) = [_, 120,  95,  70]]
        c7 = F.leaky_relu(self.conv7(c6))  # np.shape(c7) = [_, 140,  94,  35]]
        c8 = F.leaky_relu(self.conv8(c7))  # np.shape(c8) = [_, 160,  47,  34]]
        c9 = F.leaky_relu(self.conv9(c8)) # np.shape(c9) = [_, 200,  46,  17]]
        #c9 = torch.sigmoid(self.conv9(c8)) # np.shape(c9) = [_, 200,  46,  17]]
        #
        # About the activation function of the last layer:
        # sigmoid:
        #   The last layer passes through a sigmoid activation funtion to ensure that the values in feature map are in the range (0, 1)
        #   It generally takes longer to train the model when using the sigmoid activation funtion.  Set the learning rate to 1e-4.
        #   It might take a few epochs to converge.
        # leaky_relu:
        #   By using leaky_relu as activation function, it is easier to train the model.  Set the learning rate to 1e-3
        #   It converges quicker by using leaky_relu, but the the feature map values could be very large.  (use historgram to check)
        #
        # save the featureMatrix, which is the input for the deconvolution network
        # self.featureMatrix.numel() = _ * 156400
        self.featureMatrix = c9  
        #
        return self.featureMatrix


class Classifier(FeatureExtraction):
    """
    A fully connected network taking feature matrix as input and producing classifications
    """

    def __init__(self):
        super(Classifier, self).__init__()
        #
        # for Linear module weight and bias values initialization, please refer to the pytorch document
        #
        self.fc1 = nn.Linear(200*46*17, 300)
        #
        self.fc2 = nn.Linear(300, 100)
        #
        self.fc3 = nn.Linear(100, 10)


    def forward(self, x):
        fm = super().forward(x)
        fm = fm.view(-1, 200*46*17)
        #
        x1 = F.leaky_relu(self.fc1(fm))
        x2 = F.leaky_relu(self.fc2(x1))
        x3 = F.leaky_relu(self.fc3(x2))
        #x3 = self.fc3(x2)
        #
        return x3


def weights_init(m):
    classname = m.__class__.__name__
    print("module: ", classname)
    #
    if classname.find('Conv') != -1:
        (row, col) = m.kernel_size
        inC = m.in_channels
        stdev = math.sqrt(1/(inC*row*col))*0.9
        m.weight.data.normal_(0.0, stdev)
        print("for Conv modules, use customized initial weights, normal distribution: (0.0, ", stdev, ")")
    elif classname.find('Linear') != -1:
        print("for Linear modules, use the default initialization values")


#----------------------------------------------------------------------------------


from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class AudioDataSet(Dataset):

    def __init__(self, folderPath, fileNames, window, dataCnt=5000):
        """
        folderPath: full file sytem path to the folder containing the training data files
        fileNames:  a  list of file names to load as training data
        window:     the number of frames to form a training data piece
        """
        #
        self.folderPath = folderPath
        self.fileNames  = fileNames
        self.dataClsCnt = len(fileNames)  
        self.window     = window
        #
        self.audioSamples = np.array([None  for _ in range(0, self.dataClsCnt)])
        #
        self.freqBandCnt=400   # number of frequency bands to produce from fast Fourier transform 
        (self.frame_size, self.frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds
        #
        for i in range(0, self.dataClsCnt):
            filePathName = self.folderPath + self.fileNames[i] 
            #
            print("load file: ", filePathName)
            channel, filtered, frqcyBnd, sample_rate = FourierTransform.FFT(filePathName, self.freqBandCnt, self.frame_size, self.frame_stride, duration=0, emphasize=False)
            #
            self.audioSamples[i] = torch.as_tensor(filtered, dtype=torch.float)
        #
        self.dataCnt = dataCnt

    # Override to give PyTorch access to any data on the dataset
    def __getitem__(self, index):
        #
        classId = index % self.dataClsCnt
        #
        filtered = self.audioSamples[classId]
        #
        # randomly select a audio segment from the given category classId
        (_, _, N) = np.shape(filtered)
        i0 = np.random.randint(0, N-self.window)
        i1 = i0+window
        inMatrix = filtered[:, :, i0:i1]
        #
        return inMatrix, classId

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.dataCnt


#----------------------------------------------------------------------------------


from torch.utils.data import Dataset, DataLoader

def trainClassifier(classifier, training_data_set, batch_size=15, epochCnt=1):
    #
    training_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    learning_rate = 0.00001
    momentum = 0.9
    #
    #optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)   # create a stochastic gradient descent optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
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
            if (idx+1)%10==0 or idx==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, idx * len(input_batch), len(training_loader.dataset),
                               100. * idx / len(training_loader), loss.data))
            idx += 1
        #
    #


#----------------------------------------------------------------------------------
# train the classifier
        
classifier = Classifier().cuda()
classifier.apply(weights_init)

# preparing the training data
"""
folderPath = "C:\\Data\\WAV\\Environment\\"
fileNames = ["crowd.wav", "highway.wav", "rain.wav", "violin.wav", "piano.wav", "beach.wav", "birdChirping.wav", "tedTalk.wav"]
classCnt = 10  # The classifier produces 10 outputs
window = 300
"""

folderPath = 'C:\\Data\\WAV\\Environment\\'
fileNames = ['speechlong.wav', 'water-rain1.wav', 'wind01.wav', 'Corvette_pass.wav', 'lawnmower.wav', 'applause7.wav', 'police_sirens.wav', 'white_noise.wav', 'steps.wav', 'techno_drum.wav']
len(fileNames)
classCnt = 10  # The classifier produces 10 outputs
window = 300

training_data_set = AudioDataSet(folderPath, fileNames, window, dataCnt=2000)
trainClassifier(classifier, training_data_set, validation_data_set)


#----------------------------------------------------------------------------------
# plot frequency profile time series for a partitular sound
fileNames = ['speechlong.wav']
sigleAudioSet = AudioDataSet(folderPath, fileNames, window, dataCnt=200)
singleAudioLoader = DataLoader(sigleAudioSet, batch_size=1, shuffle=True, num_workers=0)

inM, _ = next(iter(singleAudioLoader))
inM = np.array(inM)

high_mel = 2595 * np.log10(1+(FourierTransform.FreqBandHigh)/700)  # Calculates the highest possible mel value given the input requency range
mels = np.linspace(0, high_mel, FREQ_BAND+2) 
hzs = 700 * (np.power(10, (mels/2596))-1) 

frqcy = [None  for _ in range(0, FREQ_BAND)]
for h in range(1, len(hzs)-1):
    # Retrieves three points on current frequency channel
    tleft = hzs[h-1]
    tright = hzs[h+1]
    tmid = (tright+tleft)/2
    frqcy[h-1] = tmid

t=0
plt.plot(frqcy, inM[0, 0, :, t:(t+1)])


#----------------------------------------------------------------------------------
# to save the model
modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'Classifier_leayReLU.pt'
modelPathName = modelPath + modelName

torch.save(classifier.state_dict(), modelPathName)


#----------------------------------------------------------------------------------
# classifier testing

def testClassifier(classifier, input_batch, target_batch):
    output_batch = classifier(input_batch)
    sm = F.softmax(output_batch, dim=1)
    #
    dataCnt=len(output_batch)
    successCnt=0
    for i in range(0, dataCnt):
        print("target: ", target_batch[i])
        print("output: ", sm[i])
        #
        if sm[i][target_batch[i]] >= 0.95:
            successCnt += 1
    #
    print("success ratio = ", successCnt/dataCnt)


# to load the model
modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'Classifier_sigmoid_feature.pt'
modelPathName = modelPath + modelName

classifier = Classifier().cuda()
classifier.load_state_dict(torch.load(modelPathName))
classifier.eval()

folderPath = 'C:\\Data\\WAV\\Environment\\'
fileNames = ['voice_whitenoise(voiceamp=2).wav']

testing_data_set = AudioDataSet(folderPath, fileNames, window, dataCnt=500)
testing_loader = DataLoader(testing_data_set, batch_size=10, shuffle=True, num_workers=0)

input_batch, target_batch = next(iter(testing_loader))
input_batch  = input_batch.to('cuda')
target_batch = target_batch.to('cuda')

output_batch = classifier(input_batch)
output_batch
F.softmax(output_batch)

testClassifier(classifier, input_batch, target_batch)

        
        
#######################################################################################


class FeatureTranspose(nn.Module):
    """
    A transposed CNN network (deconvolution network) to reconstruct the input
    """
    
    def __init__(self):
        super(FeatureTranspose, self).__init__()
        #
        self.tconv9 = nn.ConvTranspose2d(200, 160, kernel_size=( 2, 2), stride=(1, 2))
        #
        self.tconv8 = nn.ConvTranspose2d(160, 140, kernel_size=( 2, 2), stride=(2, 1))
        #
        self.tconv7 = nn.ConvTranspose2d(140, 120, kernel_size=( 2, 2), stride=(1, 2))
        #
        self.tconv6 = nn.ConvTranspose2d(120, 120, kernel_size=( 3, 3), stride=1)
        #
        self.tconv5 = nn.ConvTranspose2d(120, 100, kernel_size=( 2, 2), stride=2)
        #
        self.tconv4 = nn.ConvTranspose2d(100, 100, kernel_size=( 3, 3), stride=1)
        #
        self.tconv3 = nn.ConvTranspose2d(100, 100, kernel_size=( 3, 3), stride=1)
        #
        self.tconv2 = nn.ConvTranspose2d(100, 100, kernel_size=( 3 ,3), stride=1)
        #
        self.tconv1 = nn.ConvTranspose2d(100,  60, kernel_size=( 4, 4), stride=2)
        #
        # the input to this layer as 60 channels; they will be divided into three parts with sizes: 10, 10, 40
        self.tinc21 = nn.ConvTranspose2d( 10,  48, kernel_size=(25,  1), stride=1)
        self.tinc22 = nn.ConvTranspose2d( 10,  48, kernel_size=( 1, 25), stride=1)
        self.tinc23 = nn.ConvTranspose2d( 40,  48, kernel_size=( 5,  5), stride=1)
        #
        # the input to this layer as 48 channels; they will be divided into three parts with sizes: 10, 10, 14, 14
        self.tinc11 = nn.ConvTranspose2d( 10,  1, kernel_size=(45, 1), stride=1)
        self.tinc12 = nn.ConvTranspose2d( 10,  1, kernel_size=(1, 45), stride=1)
        self.tinc13 = nn.ConvTranspose2d( 14,  1, kernel_size=(15, 3), stride=1)
        self.tinc14 = nn.ConvTranspose2d( 14,  1, kernel_size=( 5, 5), stride=1)
        #
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # input shape:  np.shape(x) = [_, 200,  46,  17]
        #
        tc9 = F.leaky_relu(self.tconv9(x))    # np.shape(x)   = [_, 200,  46,  17]]
        tc8 = F.leaky_relu(self.tconv8(tc9))  # np.shape(tc9) = [_, 160,  47,  34]]
        tc7 = F.leaky_relu(self.tconv7(tc8))  # np.shape(tc8) = [_, 140,  94,  35]]
        tc6 = F.leaky_relu(self.tconv6(tc7))  # np.shape(tc7) = [_, 120,  95,  70]]
        tc5 = F.leaky_relu(self.tconv5(tc6))  # np.shape(tc6) = [_, 120,  97,  72]]
        tc4 = F.leaky_relu(self.tconv4(tc5))  # np.shape(tc5) = [_, 100, 194, 144]]
        tc3 = F.leaky_relu(self.tconv3(tc4))  # np.shape(tc4) = [_, 100, 196, 146]]
        tc2 = F.leaky_relu(self.tconv2(tc3))  # np.shape(tc3) = [_, 100, 198, 148]]
        tc1 = F.leaky_relu(self.tconv1(tc2))  # np.shape(tc2) = [_, 100, 200, 150]]
        #
        # np.shape(tc1) = [_, 60,  402, 302]]
        #
        (_, _, row, col) = np.shape(tc1)
        tx2 = tc1[:,:, 1:row-1, 1:col-1]
        #
        # split the output for inception pass
        tx21, tx22, tx23 = torch.split(tx2, (10, 10, 40), dim=1)
        #
        # 2nd inception layer
        p21 = self.tinc21(tx21)
        (_, _, row, col) = np.shape(p21)
        p21_ = p21[:,:,12:row-12,:]
        #    
        p22 = self.tinc22(tx22)
        (_, _, row, col) = np.shape(p22)
        p22_ = p22[:,:,:,12:col-12]
        #
        p23 = self.tinc23(tx23)
        (_, _, row, col) = np.shape(p23)
        p23_ = p23[:,:,2:row-2,2:col-2]
        #
        tx1 = F.leaky_relu(torch.add(torch.add(p21_, p22_), p23_))
        #
        # split the output for inception pass
        tx11, tx12, tx13, tx14 = torch.split(tx1, (10, 10, 14, 14), dim=1)
        #
        # 1st inception layer
        p11 = self.tinc11(tx11)
        (_, _, row, col) = np.shape(p11)
        p11_ = p11[:,:,22:row-22,:]
        #
        p12 = self.tinc12(tx12)
        (_, _, row, col) = np.shape(p12)
        p12_ = p12[:,:,:,22:col-22]
        #
        p13 = self.tinc13(tx13)
        (_, _, row, col) = np.shape(p13)
        p13_ = p13[:,:,7:row-7,1:col-1]
        #
        p14 = self.tinc14(tx14)
        (_, _, row, col) = np.shape(p14)
        p14_ = p14[:,:,2:row-2,2:col-2]
        #
        #tx0 = self.sigmoid(torch.add(torch.add(torch.add(p11_, p12_), p13_), p14_))
        tx0 = F.leaky_relu(torch.add(torch.add(torch.add(p11_, p12_), p13_), p14_))
        #
        self.transposed = tx0
        #
        return self.transposed


#----------------------------------------------------------------------------------


class AudioGenerator(nn.Module):

    def __init__(self, data_size, hidden_size):
        super(AudioGenerator, self).__init__()
        #
        self.hidden_size = hidden_size
        #
        # transpose the audio feature map (the output from the last convolution layer) into a matrix of input size
        self.trans = FeatureTranspose()
        #
        # use LSTM to translate the enlarged feature map into a time series of frequency profile
        self.lstm = nn.LSTM(input_size=data_size, hidden_size =hidden_size, num_layers=2)
        #
        # The fully connected layer that converts hidden vectors into outputs
        self.fc = nn.Linear(hidden_size, data_size)
        #
        self.transposed=None
        self.audio=None


    def forward(self, features):
        #
        self.transposed = self.trans(features)
        #
        transposed = self.transposed[0,0].t().view(1,300,400)
        
        self.transposed[0,0].t().view(1,300,400),
         #
        hidden, _ = self.lstm(transposed)
        #
        audio = self.fc(hidden)
        audio = F.sigmoid(audio)
        #
        self.audio = audio[0].t().view(1, 1, 400, 300)
        #
        return self.audio




#----------------------------------------------------------------------------------


modelObject = []
        
def weights_init_2(m):
    modelObject.append(m)
    classname = m.__class__.__name__
    print("module: ", classname)
    #
    if classname.find('Conv') != -1:
        (row, col) = m.kernel_size
        inC = m.in_channels
        stdev = math.sqrt(1/(inC*row*col))*0.8
        m.weight.data.normal_(0.0, stdev)
        print("for Conv modules, use customized initial weights, normal distribution: (0.0, ", stdev, ")")
    elif classname.find('Linear') != -1:
        print("for Linear modules, use the default initialization values")


from torch.utils.data import Dataset, DataLoader

def trainAudioGenerator(generator, training_data_set, classifierModelPathName, batch_size=5, epochCnt=1):
    #
    # load the trained classifier neural network
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load(classifierModelPathName))
    classifier.eval()
    #
    training_loader = DataLoader(training_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    learning_rate = 1e-5
    momentum = 0.9
    #
    #optimizer = torch.optim.SGD(generator.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    #lossFunc = nn.MSELoss() # create a mean squared loss function to calculate the difference between output and target
    #
    # to change the optimizer learning rate
    for opt in optimizer.param_groups: opt['lr']=1e-7
    #
    #lossFunc = nn.L1Loss()
    loss_function = nn.MSELoss()
    #
    for epoch in range(0, epochCnt):
        idx=0
        cnt=0
        #
        for inputs, labels in training_loader:   # to iterate: inputs, labels = next(iter(training_loader))
            #
            inputs = inputs * (inputs>1e-3).float()
            inputs = inputs.to('cuda')
            #
            # pass the data through the classifier first
            outputs = classifier(inputs)
            sm = F.softmax(outputs, dim=1)
            #
            # loop through the result and pick only the correctly classified ones for training the generator
            candidates = []
            for i in range(0, len(inputs)): 
                if sm[i][0] >= 0.95:    # voice has classification id of 0
                    candidates.append(True)
                else:
                    candidates.append(False)
            #
            cnt += len(candidates)
            #
            if cnt>0:
                optimizer.zero_grad()
                #
                features = classifier.featureMatrix
                features = features * (features>0).float()
                audios = generator(features[candidates])
                loss = lossFunc(audios, inputs[candidates])
                loss.backward()
                # run gradient descent based on the gradients calculated from the backward() function
                optimizer.step()    
            #
            if (idx+1)%50==0 or idx==0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, cnt, len(training_loader.dataset), loss.data))
            #
            idx += 1
        #
    #


#----------------------------------------------------------------------------------

modelPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
modelName = 'Classifier_leakyReLU_feature.pt'
classifierModelPathName = modelPath + modelName

folderPath = 'C:\\Data\\WAV\\Environment\\'
fileNames = ['speechlong.wav']
len(fileNames)
classCnt = 10  # The classifier produces 10 outputs
window = 300

# train the generator
training_data_set = AudioDataSet(folderPath, fileNames, window, dataCnt=500)

generator=AudioGenerator(400, 400).cuda()
        




trainAudioGenerator(generator, training_data_set, classifierModelPathName, batch_size=1, epochCnt=1):



########################################################################################

import time


# testing on GPU ---------------------------------------------------------------
fe = FeatureExtract().cuda()
x = torch.randn((1, 1, FREQ_BAND, SOUND_WINDOW)).cuda()

s = time.time()
featureMatrix = fe.forward(x)
e = time.time()
print(e-s)

cls = Classification().cuda()

s = time.time()
Classification = cls.forward(x)
e = time.time()
print(e-s)

audioGen = AudioGenerator().cuda()

audio = audioGen.forward(featureMatrix)

sourceTensor.clone().

# testing on CPU ---------------------------------------------------------------
fe_ = FeatureExtract()
x_ = torch.randn((1, 1, FREQ_BAND, SOUND_WINDOW))

s = time.time()
cnt_ = fe_(x_)
e = time.time()
print(e-s)


for param in fe.parameters():
    print(type(param.data), param.size())

for param in audioGen.parameters():
    print(type(param.data), param.size())











