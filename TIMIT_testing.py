import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from TIMIT import FeatureExtraction, Classifier, AudioGenerator
from TIMIT import TIMIT_PHN_reader
from TIMIT import phoncecode, phonedict

from FourierTransform import FourierTransform



#---------------------------------------------------------------------------------------------------------
# load the classifier
classifierPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
classifierName = 'Classifier_Inception_TIMIT_ALL.pt'
classifierModelPathName = classifierPath + classifierName
# load the trained classifier neural network
classifier = Classifier()#.cuda()
classifier.load_state_dict(torch.load(classifierModelPathName))
classifier.eval()


# load the softmask generator
softmaskPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
softmaskName = 'SoftMask_Subtraction_Inception_TIMIT_ALL.pt'
softmaskModelPathName = softmaskPath + softmaskName
# load the trained classifier neural network
softmask = AudioGenerator()#.cuda()
softmask.load_state_dict(torch.load(softmaskModelPathName))
softmask.eval()



#---------------------------------------------------------------------------------------------------------

freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds


# load the background white noise file
noiseFilePathName = 'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2_16K' +'.wav'
#
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(noiseFilePathName, 0, frame_size, frame_stride, duration=4, emphasize=False)
spect_noise = torch.as_tensor(spectrogram, dtype=torch.float)
spect_noise /= 25  # The noise was scaled down in amplitude when it was mixed with the pure voice.  It needs to be scaled down by the same scale.
(_, _, frameCnt_noise) = np.shape(spect_noise)


# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR1\\MTRR0\\"
audioFile = "SI918.WAV"
#
#pathName = "C:\\Data\\TIMIT\\Data\\TEST\\DR1\\FJEM0\\"
#audioFile = "SX364.WAV"
#
# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TEST\\DR6\\MRJR0\\"
audioFile = "SI2313.WAV"
#

wavfileName = audioFile + ".wav" 
wavFilePathName = pathName + wavfileName
channel, spect_voice, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
spect_voice = torch.as_tensor(spect_voice, dtype=torch.float)
#
(_, _, frameCnt_voice) = np.shape(spect_voice)


# load a sample noisy audio file
wavfileName = audioFile + ".noise.wav" 
wavFilePathName = pathName + wavfileName
#
channel, spect_mixed, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
spect_mixed = torch.as_tensor(spect_mixed, dtype=torch.float)
#
(_, _, frameCnt_mixed) = np.shape(spect_mixed)

# make them to have the same size
frameCnt = min(frameCnt_noise, frameCnt_voice, frameCnt_mixed)
spect_noise = spect_noise[:,:,0:frameCnt]
spect_voice = spect_voice[:,:,0:frameCnt]
spect_mixed = spect_mixed[:,:,0:frameCnt]

MARGIN=15
init = True
spect_cleaned = None




def classifierInput(spectrogram, fIdx):
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

           
def classifierToPhone(classOut):
    softmax = F.softmax(classOut, dim=1)
    vlu, idx = torch.max(softmax[0], 0)
    phone = phonecode[idx-1][1]
    return phone
        


def voice_enhancement(classifier, softmask, spect_mixed, frameCnt, MARGIN):
    frameSeq=[]
    #
    # add the margin frames on the left; these frames are not going to be 'cleaned'.
    for i in range(0, MARGIN): frameSeq.append(spect_mixed[:,:,i:i+1])
    #
    # run each frame through the generator and cumulate the generated freq/amp file
    for f in range(MARGIN, frameCnt-MARGIN):
        #
        inMatrix_mixed = classifierInput(spect_mixed, f)
        inMatrix_mixed = inMatrix_mixed.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
        #inMatrix_mixed = inMatrix_mixed.to('cuda')
        #
        output_mixed = classifier(inMatrix_mixed)
        fm = classifier.featureMatrix
        mask = softmask(fm)
        #
        #mask = mask.cpu()
        mask = torch.squeeze(mask, dim=0)
        #
        frame_mixed = spect_mixed[:, :, f:f+1]
        #frame_mixed = frame_mixed.to('cuda')
        frame_cleaned = frame_mixed - mask
        frameSeq.append(frame_cleaned)
        #
        print("process frame: ", f, "/", frameCnt)
        #torch.cuda.empty_cache()
        #print("cuda cached memory= ", torch.cuda.memory_cached())
    #
    #
    # add the margin frames on the right; these frames are not going to be 'cleaned'.
    # total number of frames are exactly the same as the noisy frame sequence.
    for i in range(frameCnt-MARGIN, frameCnt): frameSeq.append(spect_mixed[:,:,i:i+1])
    #
    return frameSeq



def spect_quality(spect_cleaned, spect_voice, spect_noise):
    noise = spect_cleaned - spect_voice
    noise[noise<0] = 0
    #soundWAVUtil.spectrogramHeatmap(noise.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
    #
    n = torch.sum(noise)
    N = torch.sum(spect_noise)
    NDR = n/N
    #
    voice = spect_cleaned - spect_noise
    voice[voice<0] = 0
    #soundWAVUtil.spectrogramHeatmap(voice.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
    #
    V2 = torch.sum(spect_voice*spect_voice)
    diff = voice - spect_voice
    sqr = torch.sum(diff*diff)
    VDR = torch.sqrt(sqr/V2)
    #
    v = torch.sum(voice)
    SNR = v/n
    #
    return NDR, VDR, SNR, N, n, V, v



def phoneme_classify(classifier, spect_mixed, frameCnt, MARGIN, frm_phn_xIdx):
    #
    correct = 0
    wrong   = 0
    # run each frame through the generator and cumulate the generated freq/amp file
    for f in range(MARGIN, frameCnt-MARGIN):
        #
        inMatrix = classifierInput(spect_mixed, f)
        inMatrix = inMatrix.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
        #inMatrix_noise = inMatrix_noise.to('cuda')
        #
        output = classifier(inMatrix)
        #
        sm = F.softmax(output, dim=1)
        #
        v, i = torch.max(sm[0], 0)
        if  frm_phn_xIdx[f][2] == 1:
            phn = frm_phn_xIdx[f][4][0]
            classId = phonedict[phn]
            if classId == (i):
                correct += 1
            else:
                wrong += 1
            #
        #
        ratio = correct/(correct+wrong)
        print("frame =", f, ";   correctly classified ratio = ", ratio)

    #
    ratio = correct/(correct+wrong)
    print("correctly classified ratio = ", ratio)
    #
    return ratio



frameSeq = None
frameSeq = voice_enhancement(classifier, softmask, spect_mixed, frameCnt, MARGIN)
frameSeq = voice_enhancement(classifier, softmask, spect_cleaned, frameCnt, MARGIN)

# concatenate all frames into a pytorch tensor
spect_cleaned = torch.cat(frameSeq, dim=2)
#
torch.save(spect_cleaned, path+spectName)
torch.save(spect_cleaned, 'spect_cleaned.tensor')

# load a previously saved spectrogram
spect_mixed = torch.load('spect_cleaned5.tensor')


min=torch.randn(np.shape(spect_voice)).fill_(1e-6)
lv=(spect_voice-min).log10()
soundWAVUtil.spectrogramHeatmap(lv.detach()[0], frqcyRange=[0, 150])
ln=(spect_noise-min).log10()
soundWAVUtil.spectrogramHeatmap(ln.detach()[0], frqcyRange=[0, 150])
lm=(spect_mixed-min).log10()
soundWAVUtil.spectrogramHeatmap(lm.detach()[0], frqcyRange=[0, 150])

spect_cleaned3 = torch.load('spect_cleaned3.tensor')
spect=spectrogram.clone()
spect[spectrogram<0] = 1.1e-6
lc=(spect-min).log10()
soundWAVUtil.spectrogramHeatmap(lc.detach()[0], frqcyRange=[0, 150])


l=spectrogram.log10()
soundWAVUtil.spectrogramHeatmap(l.detach()[0], frqcyRange=[0, 150])
soundWAVUtil.spectrogramHeatmap(spectrogram.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])

 
soundWAVUtil.spectrogramHeatmap(spect_noise.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_voice.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_mixed.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
#
soundWAVUtil.spectrogramHeatmap(spect_cleaned.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned1.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned2.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned3.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])


phoneme_classify(classifier, spect_voice, frameCnt, MARGIN, frm_phn_xIdx)


########################################################################################################
########################################################################################################
########################################################################################################




for t in range(0, 5):
    #
    input_batch, target_batch = next(iter(training_loader))
    #
    input_batch  = input_batch.to('cuda')    # the shape of input_batch = (batch_size, 1, FREQ_BAND, SOUND_WINDOW)
    target_batch = target_batch.to('cuda')   # the shape of target_batch = (batch_size)
    #
    output_batch = classifier(input_batch)
    sm = F.softmax(output_batch, dim=1)
    #
    total1 = 0
    total2 = 0
    success_rate1 = 0
    success_rate2 = 0
    #
    correct=[]
    wrong=[]
    for t in range(0, 50):
        v, i = torch.max(sm[t], 0)
        if target_batch[t]==i:
            correct.append(t)
        else:
            wrong.append(t)
        #
        total1 += 1
    #
    success_rate1 += len(correct)
    print(len(correct), len(wrong))
    #    
    correct=[]
    wrong=[]
    for t in range(0, 50):
        v, i = torch.topk(sm[t], 2) 
        if len((i==target_batch[t]).nonzero())>0:
            correct.append(t)
        else:
            wrong.append(t)
        #
        total2 += 1
    #
    success_rate2 += len(correct)
    print(len(correct), len(wrong))
    #
    print(success_rate1/total1, success_rate2/total2)


#----------------------------------------------------------------------------------------------

idx=92

print("original and fitted amplitutde//frequency:")          
x = [i  for i in range(0, 240)]
plt.plot(x, inputs[candidates][:, 0:1, :, 5:6][idx][0,:,0].cpu().detach().numpy())
plt.plot(x, audios[idx][0,:,0].cpu().detach().numpy())
plt.show()            

print("fitted amplitutde//frequency:")   
x = [i  for i in range(0, 240)]
plt.plot(x, audios[idx][0,:,0].cpu().detach().numpy())
plt.show()



#----------------------------------------------------------------------------------------------

freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds

# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR8\\MBCG0\\"
wavfileName = "SI957.WAV.wav" 
wavFilePathName = pathName + wavfileName
#
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
#
(_, _, frameCnt) = np.shape(spectrogram)
init = True
spect=None
#
spectrogram = torch.as_tensor(spectrogram, dtype=torch.float)



# run each fram through the generator and cumulate the generated freq/amp file
for f in range(15, frameCnt-15):
    #
    inMatrix = classifierInput(spectrogram, f)
    inMatrix = inMatrix.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output = classifier(inMatrix)
    fm = classifier.featureMatrix
    audio = generator(fm)
    #
    audio = audio.squeeze(dim=0)
    if init:
        spect=audio
        init=False
    else:
        spect = torch.cat((spect, audio), dim=2)


# generate a wave file
generatedWavFilePath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
generatedWavFilePathName = generatedWavFilePath + wavfileName
FourierTransform.playSound(frame_size, frame_stride, spect_cleaned, None, frqcyBnd, sample_rate, 0, 3, generatedWavFilePathName)

    
#----------------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------------    
    
idx=11

print("weight and mask comparison:")          
x = [i  for i in range(0, 240)]
plt.plot(x, weight[idx, 0, :, 0].cpu().detach().numpy())
plt.plot(x, mask[idx, 0, :, 0].cpu().detach().numpy())
plt.show()            

    
idx=36
idx = np.random.randint(0, 200)
print("weight and mask comparison:", idx)          
x = [i  for i in range(0, 240)]
plt.plot(x, original[idx, 0, :, 0].cpu().detach().numpy())
plt.plot(x, noisy[idx, 0, :, 0].cpu().detach().numpy())
plt.show()
plt.plot(x, diff[idx, 0, :, 0].cpu().detach().numpy())
plt.plot(x, mask[idx, 0, :, 0].cpu().detach().numpy())
plt.show()   
plt.plot(x, original[idx, 0, :, 0].cpu().detach().numpy())
plt.plot(x, noisy[idx, 0, :, 0].cpu().detach().numpy()-mask[idx, 0, :, 0].cpu().detach().numpy())
plt.show()
#----------------------------------------------------------------------------------------------------    



soundWAVUtil.spectrogramHeatmap(spect_cleaned.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])

soundWAVUtil.spectrogramHeatmap(torch.load('spect_cleaned6.tensor').detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])

spect_cleaned1 = spect_cleaned 
spect_cleaned2 = spect_cleaned
spect_cleaned3 = spect_cleaned



s0 = spect_mixed - spect_noise
soundWAVUtil.spectrogramHeatmap(s0.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])

s1 = spect_mixed - spect_voice
soundWAVUtil.spectrogramHeatmap(s1.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])


soundWAVUtil.spectrogramHeatmap(spect_noise0.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_noise.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])


SNR, NDR, VDR, N, n, V, v = spect_cleaned(spect_cleaned, spect_voice, spect_noise)
noiseSize1, voiceDiff1 = spect_distance(spect_cleaned, spect_voice, spect_noise)
noiseSize2, voiceDiff2 = spect_distance(spect_cleaned2, spect_voice, spect_noise)

voiceDiff1- voiceDiff0


torch.sum(spect_noise*spect_noise)
torch.sum(spect_voice*spect_voice)






path = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
spectName = 'spect_cleaned7.pt'
torch.save(spect_cleaned, path+spectName)
spect_cleaned=None
spect_noise = torch.load(path+spectName)

spect_cleaned = torch.load(path+'spect_cleaned6.pt')


pO=[]
pN=[]
pC=[]

for f in range(MARGIN, frameCnt-MARGIN):
    #
    inMatrix_origin = classifierInput(spect_voice, f)
    inMatrix_origin = inMatrix_origin.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_origin = classifier(inMatrix_origin)
    phone_origin = classifierToPhone(output_origin)
    pO.append(phone_origin)
    #
    #
    inMatrix_noise = classifierInput(spect_noise0, f)
    inMatrix_noise = inMatrix_noise.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_noise = classifier(inMatrix_noise)
    phone_noise = classifierToPhone(output_noise)
    pN.append(phone_noise)
    #
    #
    inMatrix_cleaned = classifierInput(spect_cleaned, f)
    inMatrix_cleaned = inMatrix_cleaned.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_cleaned = classifier(inMatrix_cleaned)
    phone_cleaned = classifierToPhone(output_cleaned)
    pC.append(phone_cleaned)
    #
    #
    print("process frame: ", f, "/", frameCnt)
#




pO = np.array(pO)
pN = np.array(pN)
pC = np.array(pC)

ON=(pO==pN)
OC=(pO==pC)

print(OC.sum(), ON.sum())

#----------------------------------------------------------------------------------------------------    


L1 = 0
L2 = 0
for f in range(15, frameCnt-15):
    origin = spectrogram_original[:, :, f:f+1]
    wNoise = spectrogram[:, :, f:f+1]
    masked = spect_cpu[:, :, f-15:f-15+1]
    #
    l1 = scipy.spatial.distance.cdist(origin[0], wNoise[0]).sum()
    l2 = scipy.spatial.distance.cdist(origin[0], masked[0]).sum()
    #
    L1 += l1
    L2 += l2
    #
    print("frame improvement=", 1-l2/l1, "L1=", L1, "L2=", L2)
#
    

# generate a wave file
generatedWavFilePath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
wavfileName = "SI918.WAV.cleaned.wav"
generatedWavFilePathName = generatedWavFilePath + wavfileName
FourierTransform.playSound(frame_size, frame_stride, spect_cleaned, phasegram, frqcyBnd, sample_rate, 0, 3.5, generatedWavFilePathName)


#----------------------------------------------------------------------------------------------------  

# run each fram through the generator and cumulate the generated freq/amp file
for f in range(15, frameCnt-15):
    #
    inMatrix_noise = classifierInput(spect_noise, f)
    inMatrix_noise = inMatrix_noise.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_noise = classifier(inMatrix_noise)
    #
    fm = classifier.featureMatrix
    mask = softmask(fm)
    #
    mask = torch.squeeze(mask, dim=0)
    frame_noise = spect_noise[:, :, f:f+1]
    frame_cleaned = mask * frame_noise
    #
    if init:
        spect=frame_cleaned
        init=False
    else:
        spect_cleaned = torch.cat((spect_cleaned, frame_cleaned), dim=2)
    #   
    torch.cuda.empty_cache()
    print("cuda cached memory= ", torch.cuda.memory_cached())
#


    

spect_cpu = spect.cpu().detach()
spect_cpu = np.array(spect_cpu)

#----------------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------------    


x = np.arange(5)
ft  = scipy.fftpack.fft(x)
ift = scipy.fftpack.ifft(ft)
np.allclose(ift, x, atol=1e-15)  # within numerical accuracy.




