import pycuda.driver as cuda

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


import scipy.io.wavfile
import csv
import math

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


#from FourierTransform import FourierTransform



# the following functions test GPU availability    
torch.cuda.is_available()
cuda.init()

cuda.Device.count()                # number of GPU(s)
id = torch.cuda.current_device()   # Get Id of default device
cuda.Device(id).name()             # Get the name for the device "Id"

torch.cuda.memory_allocated()  # the amount of GPU memory allocated   
torch.cuda.memory_cached()     # the amount of GPU memory cached   

torch.cuda.empty_cache()  #release all the GPU memory cache that can be freed.




class FeatureExtraction(nn.Module):
    """
    This is a sequential model starting with two inception layers at the front, 
    followed by five convolutional layers of. 
    The output of the last layer is supposed to contain sufficient features to facilitate classification.
    The output of the last layer can be used as the input to a transposed CNN network (deconvolutional network) to
    reconstruct the input.
    """

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        #
        # input size = (_, 3, 240, 11)
        self.inc11 = nn.Conv2d(  3,  10, kernel_size=(45, 1), stride=(1, 1), padding=(22, 0))
        self.inc12 = nn.Conv2d(  3,  10, kernel_size=( 1, 5), stride=(1, 1), padding=( 0, 2))
        self.inc13 = nn.Conv2d(  3,  10, kernel_size=( 5, 5), stride=(1, 1), padding=( 2, 2))
        #
        self.inc21 = nn.Conv2d( 30,  20, kernel_size=(25, 1), stride=(1, 1), padding=(12, 0))
        self.inc22 = nn.Conv2d( 30,  20, kernel_size=( 1, 3), stride=(1, 1), padding=( 0, 1))
        self.inc23 = nn.Conv2d( 30,  20, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        '''
        # replace first two inception layers with two regular convolutional layers that produce the same number of channels 
        self.conv01= nn.Conv2d(  3,  30, kernel_size=( 5, 5), stride=(1, 1), padding=( 2, 2))
        self.conv02= nn.Conv2d( 30,  60, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        '''
        #
        # regular convolutional layers:
        self.conv1 = nn.Conv2d( 60, 100, kernel_size=( 3, 2), stride=(1, 1))
        #
        self.conv2 = nn.Conv2d(100, 120, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv3 = nn.Conv2d(120, 140, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv4 = nn.Conv2d(140, 160, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv5 = nn.Conv2d(160, 180, kernel_size=( 4 ,4), stride=(2, 2))    
        #
        self.conv6 = nn.Conv2d(180, 200, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))    
        #
        self.conv7 = nn.Conv2d(200, 220, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))    
        #
        self.conv8 = nn.Conv2d(220, 240, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv9 = nn.Conv2d(240, 260, kernel_size=( 2, 2), stride=(2, 2))   
        #
        self.conv10= nn.Conv2d(260, 280, kernel_size=( 2, 2), stride=(1, 1))   
    #


    def forward(self, x):
        #
        # input size = (_, 3, 240, 111)
        x11 = F.leaky_relu(self.inc11(x))
        x12 = F.leaky_relu(self.inc12(x))
        x13 = F.leaky_relu(self.inc13(x))
        x1  = torch.cat((x11, x12, x13), dim=1)  # np.shape(x1) = [_, 30, 240, 11]
        #
        x21 = F.leaky_relu(self.inc21(x1))
        x22 = F.leaky_relu(self.inc22(x1))  
        x23 = F.leaky_relu(self.inc23(x1)) 
        x2  = torch.cat((x21, x22, x23), dim=1)  # np.shape(x2) = [_, 60, 240, 11]
        #
        '''
        # replace first two inception layers with two regular convolutional layers that produce the same number of channels 
        x01= F.leaky_relu(self.conv01(x))
        x02= F.leaky_relu(self.conv02(x01))
        x2 = x02
        '''
        #
        c1 = F.leaky_relu(self.conv1(x2))  # np.shape(c1) = [_, 100, 238, 10]]
        c2 = F.leaky_relu(self.conv2(c1))  # np.shape(c2) = [_, 100, 240, 12]]
        c3 = F.leaky_relu(self.conv3(c2))  # np.shape(c3) = [_, 120, 118, 4]]
        c4 = F.leaky_relu(self.conv4(c3))  # np.shape(c4) = [_, 140, 118, 4]]
        c5 = F.leaky_relu(self.conv5(c4))  # np.shape(c5) = [_, 200, 58, 1]]
        c6 = F.leaky_relu(self.conv6(c5))
        c7 = F.leaky_relu(self.conv7(c6))
        c8 = F.leaky_relu(self.conv8(c7))
        c9 = F.leaky_relu(self.conv9(c8))
        c10= F.leaky_relu(self.conv10(c9))
        #c8 = torch.sigmoid(self.conv8(c7)) # np.shape(c9) = [_, 200, 58, 1]]
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
        self.featureMatrix = c10  
        #
        return self.featureMatrix
    #

    def parameterCnt(self):
        pcnt=0
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt


#----------------------------------------------------------------------------------


class Classifier(FeatureExtraction):
    """
    A fully connected network taking feature matrix as input and producing classifications
    """

    def __init__(self):
        super(Classifier, self).__init__()
        #
        # for Linear module weight and bias values initialization, please refer to the pytorch document
        #
        self.fc1 = nn.Linear(280*58*1, 100)
        #
        self.fc2 = nn.Linear(100, 80)
        #
        self.fc3 = nn.Linear(80, 65)



    def forward(self, x):
        fm = super().forward(x)
        fm = fm.view(-1, 280*58*1)
        #
        out1 = F.leaky_relu(self.fc1(fm))
        out2 = F.leaky_relu(self.fc2(out1))
        out3 = F.leaky_relu(self.fc3(out2))
        #
        return out3


    def parameterCnt(self):
        pcnt = super().parameterCnt()
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt


#----------------------------------------------------------------------------------

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
#----------------------------------------------------------------------------------


class TIMITDataSet(Dataset):

    def __init__(self, TIMITRootPath, dialects, dataFrame, phoneDict, dataCnt=5000):
        """
        TIMITRootPath:  the root path to the TIMIT repository
        dialects:  a list of dialects to load speech corpus from
        dataCnt:  total numbe of speech corpus
        """
        #
        self.TIMITRootPath = TIMITRootPath
        self.TIMITDataPath = TIMITRootPath + "Data\\"
        self.dialects  = dialects
        self.dataCnt = dataCnt 
        self.phoneDict = phoneDict
        #
        dataframePathName = TIMITRootPath + dataFrame
        dataSet = pd.read_pickle(dataframePathName)
        #
        dialectSub = (dataSet.dialect_region.isin(dialects))
        dataSet = dataSet[dialectSub]
        self.dataSet = dataSet.reset_index(drop=True)
        self.fileCnt = len(self.dataSet)
        #
        self.audios = np.array([None  for _ in range(0, self.fileCnt)])
        self.phones = np.array([None  for _ in range(0, self.fileCnt)])
        self.frmCnt = np.array([None  for _ in range(0, self.fileCnt)])
        #
        self.freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
        (self.frame_size, self.frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds
        #
        for i in range(0, self.fileCnt):
            #
            fileName = self.dataSet.loc[i].wav.replace('\\\\', '\\')
            filePathName = self.TIMITDataPath + fileName
            #
            print("file count: ", i, " loading file: ", filePathName)
            channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(filePathName, 0, self.frame_size, self.frame_stride, duration=0, emphasize=False)
            self.audios[i] = torch.as_tensor(spectrogram, dtype=torch.float)
            #
            fileName = self.dataSet.loc[i].phn.replace('\\\\', '\\')
            frameBitCnt  = int(self.frame_size*sample_rate)
            strideBitCnt = int(self.frame_stride*sample_rate)
            frm_phn_xIdx = TIMIT_PHN_reader(self.TIMITDataPath, fileName, sample_cnt, frameBitCnt, strideBitCnt)
            self.phones[i] = frm_phn_xIdx
            #
            self.frmCnt[i] = len(frm_phn_xIdx)
        #
    #


    def classifierInput(self, spectrogram, fIdx):
        #
        c0=torch.cat((spectrogram[:,:, fIdx-5 :fIdx-5+1], 
                      spectrogram[:,:, fIdx-4 :fIdx-4+1],
                      spectrogram[:,:, fIdx-3 :fIdx-3+1],
                      spectrogram[:,:, fIdx-2 :fIdx-2+1],
                      spectrogram[:,:, fIdx-1 :fIdx-1+1],
                      spectrogram[:,:, fIdx-0 :fIdx-0+1],
                      spectrogram[:,:, fIdx+1 :fIdx+1+1],
                      spectrogram[:,:, fIdx+2 :fIdx+2+1],
                      spectrogram[:,:, fIdx+3 :fIdx+3+1],
                      spectrogram[:,:, fIdx+4 :fIdx+4+1],
                      spectrogram[:,:, fIdx+5 :fIdx+5+1]), dim=2)
        #
        c1=torch.cat((spectrogram[:,:, fIdx-10:fIdx-10+1], 
                      spectrogram[:,:, fIdx-8 :fIdx-8+1],
                      spectrogram[:,:, fIdx-6 :fIdx-6+1],
                      spectrogram[:,:, fIdx-4 :fIdx-4+1],
                      spectrogram[:,:, fIdx-2 :fIdx-2+1],
                      spectrogram[:,:, fIdx-0 :fIdx-0+1],
                      spectrogram[:,:, fIdx+2 :fIdx+2+1],
                      spectrogram[:,:, fIdx+4 :fIdx+4+1],
                      spectrogram[:,:, fIdx+6 :fIdx+6+1],
                      spectrogram[:,:, fIdx+8 :fIdx+8+1],
                      spectrogram[:,:, fIdx+10:fIdx+10+1]), dim=2)
        #
        c2=torch.cat((spectrogram[:,:, fIdx-15:fIdx-15+1], 
                      spectrogram[:,:, fIdx-12:fIdx-12+1],
                      spectrogram[:,:, fIdx-9 :fIdx-9+1],
                      spectrogram[:,:, fIdx-6 :fIdx-6+1],
                      spectrogram[:,:, fIdx-3 :fIdx-3+1],
                      spectrogram[:,:, fIdx-0 :fIdx-0+1],
                      spectrogram[:,:, fIdx+3 :fIdx+3+1],
                      spectrogram[:,:, fIdx+6 :fIdx+6+1],
                      spectrogram[:,:, fIdx+9 :fIdx+9+1],
                      spectrogram[:,:, fIdx+12:fIdx+12+1],
                      spectrogram[:,:, fIdx+15:fIdx+15+1]), dim=2)
        #
        inMatrix=torch.cat((c0,c1,c2), dim=0)
        #
        return inMatrix
    #


    # Override to give PyTorch access to any data on the dataset
    def __getitem__(self, index):
        #
        # simply pick a file randomly from the audio samples
        #  
        spectrogram = None
        #
        frmCnt=0
        while frmCnt<30:
            idx = np.random.randint(0, self.fileCnt)
            spectrogram = self.audios[idx]
            (_, frqCnt, frmCnt) = np.shape(spectrogram)
        #
        frmPhnX = self.phones[idx]
        #
        fIdx=0
        phnCnt=9
        MARGIN=15
        while phnCnt>1:
            fIdx = np.random.randint(MARGIN, frmCnt-MARGIN)
            if len(frmPhnX)>fIdx:
                phnCnt = frmPhnX[fIdx][2]
        #
        inMatrix = self.classifierInput(spectrogram, fIdx)
        #
        phn = frmPhnX[fIdx][4][0]
        classId = self.phoneDict[phn]
        #
        return inMatrix, classId

    def setDataCnt(self, dataCnt):
        self.dataCnt = dataCnt

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.dataCnt

    def setPhoneDict(self, phoneDict):
        self.phoneDict = phoneDict



#----------------------------------------------------------------------------------


class TIMITNoisyDataSet(TIMITDataSet):

    def __init__(self, TIMITRootPath, dialects, phoneDict, dataCnt=5000):
        super(TIMITNoisyDataSet, self).__init__(TIMITRootPath, dialects, phoneDict, dataCnt)
        """
        TIMITRootPath:  the root path to the TIMIT repository
        dialects:  a list of dialects to load speech corpus from
        dataCnt:  total numbe of speech corpus
        """
        #
        self.noisy = np.array([None  for _ in range(0, self.fileCnt)])
        #
        for i in range(0, self.fileCnt):
            #
            fileName = self.dataSet.loc[i].wav.replace('\\\\', '\\')
            l = len(fileName)
            fileName = fileName[0:l-4] + ".noise" + ".wav"
            filePathName = self.TIMITDataPath + fileName
            #
            print("file count: ", i, " loading file: ", filePathName)
            channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt, fft= FourierTransform.FFT(filePathName, 0, self.frame_size, self.frame_stride, duration=0, emphasize=False)
            self.noisy[i] = torch.as_tensor(spectrogram, dtype=torch.float)
        #
    #


    # Override to give PyTorch access to any data on the dataset
    def __getitem__(self, index):
        #
        idx = np.random.randint(0, self.fileCnt)
        #
        frmCnt=0
        while frmCnt<30:
            idx = np.random.randint(0, self.fileCnt)
            spectrogram = self.audios[idx]
            (_, frqCnt, frmCnt) = np.shape(spectrogram)
        #
        frmPhnX  = self.phones[idx]
        original = self.audios[idx]
        noisy    = self.noisy[idx]
        #
        (_, frqCnt, frmCnt) = np.shape(original)
        #
        found = False
        fIdx=0
        MARGIN=15
        while not found:
            fIdx = np.random.randint(MARGIN, frmCnt-MARGIN)
            if len(frmPhnX)>fIdx:
                found = True
        #
        phn = frmPhnX[fIdx][4][0]
        classId = self.phoneDict[phn]
        #
        rnd = np.random.randint(0, self.fileCnt)
        if rnd==0:
            return [super().classifierInput(original, fIdx), original[:,:, fIdx:fIdx+1], original[:,:, fIdx:fIdx+1]], classId
        else:
            return [super().classifierInput(original, fIdx), original[:,:, fIdx:fIdx+1], noisy[:,:, fIdx:fIdx+1]], classId
        #
    #



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------



class FeatureTranspose(nn.Module):
    """
    A transposed CNN network (deconvolution network) to reconstruct the input
    """
    
    def __init__(self):
        super(FeatureTranspose, self).__init__()
        #
        self.tinc01 = nn.ConvTranspose2d(280,  70, kernel_size=( 1, 1), stride=1, padding=(0, 0))
        self.tinc02 = nn.ConvTranspose2d(280,  70, kernel_size=( 5, 1), stride=1, padding=(2, 0))
        self.tinc03 = nn.ConvTranspose2d(280,  70, kernel_size=( 9, 1), stride=1, padding=(4, 0))
        self.tinc04 = nn.ConvTranspose2d(280,  70, kernel_size=(13, 1), stride=1, padding=(6, 0))
        #
        self.tinc11 = nn.ConvTranspose2d(280,  80, kernel_size=( 5, 1), stride=1, padding=(0, 0)) # 80,  60-2*0,  1
        self.tinc12 = nn.ConvTranspose2d(280,  80, kernel_size=( 9, 1), stride=1, padding=(2, 0)) # 80,  64-2*2,  1
        self.tinc13 = nn.ConvTranspose2d(280,  80, kernel_size=(13, 1), stride=1, padding=(4, 0)) # 80,  68-2*4,  1
        #
        self.tinc21 = nn.ConvTranspose2d(240,  50, kernel_size=( 5, 1), stride=1, padding=(0, 0)) # 50,  66-2*0,  1
        self.tinc22 = nn.ConvTranspose2d(240,  50, kernel_size=( 7, 1), stride=1, padding=(1, 0)) # 50,  68-2*1,  1
        self.tinc23 = nn.ConvTranspose2d(240,  50, kernel_size=( 9, 1), stride=1, padding=(2, 0)) # 50,  70-2*2,  1        
        self.tinc24 = nn.ConvTranspose2d(240,  50, kernel_size=(11, 1), stride=1, padding=(3, 0)) # 50,  72-2*3,  1        
        #
        self.tconv1 = nn.ConvTranspose2d(200, 180, kernel_size=( 9, 1), stride=1)  # 180,  74,  1 
        #
        self.tconv2 = nn.ConvTranspose2d(180, 150, kernel_size=( 9, 1), stride=1)  # 150,  82,  1 
        #
        self.tconv3 = nn.ConvTranspose2d(150, 120, kernel_size=( 9, 1), stride=1)  # 120,  90,  1
        #
        self.tconv4 = nn.ConvTranspose2d(120, 100, kernel_size=( 9, 1), stride=1)  # 100,  98,  1
        #
        self.tconv5 = nn.ConvTranspose2d(100,  80, kernel_size=( 4, 1), stride=2)  #  80, 198,  1
        #
        self.tconv6 = nn.ConvTranspose2d( 80,  40, kernel_size=( 9, 1), stride=1)  #  40, 206,  1 
        #
        self.tconv7 = nn.ConvTranspose2d( 40,  20, kernel_size=( 9, 1), stride=1)  #  20, 214,  1 
        #
        self.tconv8 = nn.ConvTranspose2d( 20,  10, kernel_size=( 9, 1), stride=1)  #  10, 222,  1 
        #
        self.tconv9 = nn.ConvTranspose2d( 10,   5, kernel_size=( 9, 1), stride=1)  #   5, 230,  1 
        #
        self.tconv10= nn.ConvTranspose2d(  5,   1, kernel_size=(11, 1), stride=1)  #   1, 240,  1 
        #
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
    #


    def forward(self, x):
        #
        # input size = (_, 280, 58, 1)
        x01 = F.leaky_relu(self.tinc01(x))
        x02 = F.leaky_relu(self.tinc02(x))  
        x03 = F.leaky_relu(self.tinc03(x)) 
        x04 = F.leaky_relu(self.tinc04(x)) 
        x0  = torch.cat((x01, x02, x03, x04), dim=1)  # np.shape(x0) = [_, 280, 58, 1]
        #
        x11 = F.leaky_relu(self.tinc11(x0))
        x12 = F.leaky_relu(self.tinc12(x0))
        x13 = F.leaky_relu(self.tinc13(x0))
        x1  = torch.cat((x11, x12, x13), dim=1)  # np.shape(x1) = [_, 240, 62, 1]
        #
        x21 = F.leaky_relu(self.tinc21(x1))
        x22 = F.leaky_relu(self.tinc22(x1))  
        x23 = F.leaky_relu(self.tinc23(x1)) 
        x24 = F.leaky_relu(self.tinc24(x1)) 
        x2  = torch.cat((x21, x22, x23, x24), dim=1)  # np.shape(x2) = [_, 200, 66, 1]
        #
        #
        tc1 = F.leaky_relu(self.tconv1(x2))    
        tc2 = F.leaky_relu(self.tconv2(tc1))  
        tc3 = F.leaky_relu(self.tconv3(tc2))  
        tc4 = F.leaky_relu(self.tconv4(tc3))  
        tc5 = F.leaky_relu(self.tconv5(tc4))  
        tc6 = F.leaky_relu(self.tconv6(tc5))  
        tc7 = F.leaky_relu(self.tconv7(tc6))  
        tc8 = F.leaky_relu(self.tconv8(tc7))  
        tc9 = F.leaky_relu(self.tconv9(tc8)) 
        #
        #tc10_=self.tconv10(tc9)
        #tc10= torch.mul(torch.tanh(tc10_), F.relu(tc10_))
        #
        tc10 = self.hardtanh(self.tconv10(tc9)) 
        #tc10= F.relu(self.tconv10(tc9)) 
        #
        self.transposed = tc10
        #
        return self.transposed
    #


    def parameterCnt(self):
        pcnt=0
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt
    #
    

#----------------------------------------------------------------------------------


class AudioGenerator(FeatureTranspose):
    def __init__(self):
        super(AudioGenerator, self).__init__()
    #


    def forward(self, features):
        return super().forward(features)
    #


    def parameterCnt(self):
        pcnt = super().parameterCnt()
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt


"""
x = torch.randn(1, 280, 58, 1)
ag = AudioGenerator()
amp = ag.forward(x)

ag.parameterCnt()
"""







"""
The dialect regions are:
     dr1:  New England
     dr2:  Northern
     dr3:  North Midland
     dr4:  South Midland
     dr5:  Southern
     dr6:  New York City
     dr7:  Western
     dr8:  Army Brat (moved around)

       Dialect
      Region(dr)    #Male    #Female    Total
      ----------  --------- ---------  ----------
         1         31 (63%)  18 (27%)   49 (8%)  
         2         71 (70%)  31 (30%)  102 (16%) 
         3         79 (67%)  23 (23%)  102 (16%) 
         4         69 (69%)  31 (31%)  100 (16%) 
         5         62 (63%)  36 (37%)   98 (16%) 
         6         30 (65%)  16 (35%)   46 (7%) 
         7         74 (74%)  26 (26%)  100 (16%) 
         8         22 (67%)  11 (33%)   33 (5%)
       ------     --------- ---------  ---------- 
         8        438 (70%) 192 (30%)  630 (100%)
"""

 
 

def TIMIT_data_directory_loader(pathName, fileName):
    dataFilePathName = pathName + fileName
    #
    cols = ['dialect_region', 'speaker_id', 'file_index', 'phn', 'wav', 'txt', 'wrd']
    df = pd.DataFrame(columns=cols)
    #
    with open(dataFilePathName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                filename = row[4]
                fullPathFileName = row[6]
                #
                # find out the file name root  (file name excluding the extension)
                idx = filename.find('.')
                filename_root = filename[:idx]
                file_index = row[2] + "_" + row[3] + "_" + filename_root   # dialect_region + speaker_id + filename_root
                #
                r = (df['file_index'] == file_index)
                if not r.any():
                    rowCnt = df['file_index'].count()
                    df.loc[rowCnt] = [row[2], row[3], file_index, None, None, None, None]
                    r = (df['file_index'] == file_index)
                #
                # find out the file type (the last three characters of a file name)
                l=len(filename)
                file_type = filename[l-3:l]
                if file_type == "PHN":   # a phone file
                    df.loc[r, 'phn'] = fullPathFileName
                elif file_type == "wav":
                    df.loc[r, 'wav'] = fullPathFileName
                elif file_type == "TXT":
                    df.loc[r, 'txt'] = fullPathFileName
                elif file_type == "WRD":
                    df.loc[r, 'wrd'] = fullPathFileName
            #
            line_count += 1
        #
        print(f'Processed {line_count} lines.')
    #
    return df


# --------------------------------------------------------------------------------------------------------------------------



def GenerateTIMITDataSetDict(TIMITRoot, fileName, dataFrameFileName):
    #
    dataSet = TIMIT_data_directory_loader(TIMITRoot, fileName)
    #
    dataframeFilePathName = TIMITRoot + dataFrameFileName
    dataSet.to_pickle(dataframeFilePathName)  # save the data set to a file
    #
    dataSet = pd.read_pickle(dataframeFilePathName)



TIMITRoot = "C:\\Data\\TIMIT\\"
# GenerateTIMITDataSetDict(TIMITRoot, "train_data.csv", "train_data.df")
# GenerateTIMITDataSetDict(TIMITRoot, "test_data.csv",  "test_data.df")


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


def TIMIT_PHN_reader(filePath, phoneFileName, sample_cnt, frameBitCnt, strideBitCnt):
    """
    
    RETURN:
        frm_phn_xIdx: frame to phone cross index with the following format
        
        [frameIndex, [soundSampleIndex, soundSampleIndex + frameSize], phoneCount, percentageList, phoneList], where
        
            'phoneCount' represents the number phones the frame covers; it is an integer that's equal or greater than 1;
            'percentageList' is a list of percentages representing each phone's coverage within the frame (number of elements equal to 'phoneCount') 
            'phoneList' is a list of phones representing the phones within the frame (number of elements equal to 'phoneCount') 
    """
    #
    phoneFilePathName = filePath + phoneFileName
    phone_list = open(phoneFilePathName, 'r').read().split()
    #
    phnCnt  = int(len(phone_list)/3)
    phn_range  = np.zeros((phnCnt, 2), dtype=int)
    phn_symble = ['  '    for _ in range(0, phnCnt)]
    #
    for i in range(0, phnCnt):
        idx = i*3
        phn_range[i][0] = phone_list[idx]
        phn_range[i][1] = phone_list[idx+1]
        phn_symble[i]   = phone_list[idx+2]
    #
    # duration of each phone
    #phn_range[:,1] - phn_range[:,0]
    #
    frm_phn_xIdx = []
    frmIdx = -1
    #
    for sampleIdx in range(0, sample_cnt-frameBitCnt, strideBitCnt):
        frmIdx += 1
        pS = -1
        pE = -1
        #
        phnIdx=0
        while(phnIdx<phnCnt and pS==-1 and pE==-1):
            if phn_range[phnIdx, 0]<=sampleIdx and sampleIdx<phn_range[phnIdx, 1]:
                # found the phone index where the frame starts
                pS = phnIdx
                #
                # look for the phone index where the frame ends
                while(phnIdx<phnCnt and pE==-1):
                    if (sampleIdx+strideBitCnt-1)<phn_range[phnIdx, 1]:
                        # found the phone index where the frame ends
                        pE = phnIdx
                    else:
                        phnIdx += 1
                #
            else:
                phnIdx += 1
            #
        #
        if(pS>=0 and pE>=0):
            pct = []
            if pS==pE:
                pct.append(1.0)
            else: 
                for p in range(pS, pE+1):
                    sS = frmIdx * strideBitCnt
                    sE = sS + frameBitCnt
                    if p==pS: 
                        pct.append((phn_range[p, 1]-sS) / frameBitCnt)
                    elif p<pE:
                        pct.append((phn_range[p, 1]-phn_range[p, 0]) / frameBitCnt)
                    elif p==pE:
                        pct.append((sE-phn_range[p, 0]) / frameBitCnt)
                #
            #
            phns = []
            for p in range(pS, pE+1):
                phns.append(phn_symble[p])
            #
            frm_phn_xIdx.append([frmIdx, [sampleIdx, sampleIdx+frameBitCnt], (pE-pS+1), pct, phns])
        #
    #
    return frm_phn_xIdx


# --------------------------------------------------------------------------------------------------------------------------

"""
# testing the frame-phone cross reference

filePath = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR1\\FJSP0\\"
phoneFileName = "SX84.PHN"
waveFileName  = "SX84.WAV.wav"

(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt = FourierTransform.FFT(filePath+waveFileName, 0, frame_size, frame_stride, duration=0, emphasize=False)

frameBitCnt  = int(sample_rate*frame_size)
strideBitCnt = int(sample_rate*frame_stride)
frm_phn_xIdx = TIMIT_PHN_reader(filePath, phoneFileName, sample_cnt, frameBitCnt, strideBitCnt)
"""

# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------





# phoncode:

phonecode = [
        [1,  'b'],
        [2,  'd'],
        [3,  'g'],
        [4,  'p'],
        [5,  't'],
        [6,  'k'],
        [7,  'bcl'],
        [8,  'dcl'],
        [9,  'gcl'],
        [10, 'pcl'],
        [11, 'tcl'],
        [12, 'kcl'],  
        [13, 'dx'],
        [14, 'q'],
        
        [15, 'jh'],
        [16, 'ch'],
      
        [17,  's'],
        [18, 'sh'],
        [19,  'z'],
        [20, 'zh'],
        [21,  'f'],
        [22, 'th'],
        [23,  'v'],
        [24, 'dh'],
      
        [25,  'm'],
        [26,  'n'],
        [27, 'ng'],
        [28, 'em'],
        [29, 'en'],
        [30,'eng'],
        [31, 'nx'],
        
        [32,  'l'],
        [33,  'r'],
        [34,  'w'],
        [35,  'y'],
        [36, 'hh'],
        [37, 'hv'],
        [38, 'el'],
        
        [39, 'iy'],
        [40, 'ih'],
        [41, 'eh'],
        [42, 'ey'],
        [43, 'aa'],
        [44, 'ae'],
        [45, 'aw'],
        [46, 'ay'],
        [47, 'ah'],
        [48, 'ao'],
        [49, 'oy'],
        [50, 'ow'],
        [51, 'uh'],
        
        [52,  'uw'],
        [53,  'ux'],
        [54,  'er'],
        [55,  'ax'],
        [56,  'ix'],
        [57, 'axr'],
        [58,'ax-h'],

        [59, 'pau'],
        [60, 'epi'],
        [61, 'h#']]

"""
for i in range(0, len(phoncode)):
    print("'", phoncode[i][1], "' : ", phoncode[i][0], ",", sep='')
"""

phonedict = {  
    'b' : 1,
    'd' : 2,
    'g' : 3,
    'p' : 4,
    't' : 5,
    'k' : 6,
    'bcl' : 7,
    'dcl' : 8,
    'gcl' : 9,
    'pcl' : 10,
    'tcl' : 11,
    'kcl' : 12,
    'dx' : 13,
    'q' : 14,
    'jh' : 15,
    'ch' : 16,
    's' : 17,
    'sh' : 18,
    'z' : 19,
    'zh' : 20,
    'f' : 21,
    'th' : 22,
    'v' : 23,
    'dh' : 24,
    'm' : 25,
    'n' : 26,
    'ng' : 27,
    'em' : 28,
    'en' : 29,
    'eng' : 30,
    'nx' : 31,
    'l' : 32,
    'r' : 33,
    'w' : 34,
    'y' : 35,
    'hh' : 36,
    'hv' : 37,
    'el' : 38,
    'iy' : 39,
    'ih' : 40,
    'eh' : 41,
    'ey' : 42,
    'aa' : 43,
    'ae' : 44,
    'aw' : 45,
    'ay' : 46,
    'ah' : 47,
    'ao' : 48,
    'oy' : 49,
    'ow' : 50,
    'uh' : 51,
    'uw' : 52,
    'ux' : 53,
    'er' : 54,
    'ax' : 55,
    'ix' : 56,
    'axr' : 57,
    'ax-h' : 58,
    'pau' : 59,
    'epi' : 60,
    'h#' : 61 }

