

voiceFileName = "removenoise.wav"
voicePathName = "C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\volice\\"
voiceFilePathName = voicePathName + voiceFileName
rateV, signalV = scipy.io.wavfile.read(voiceFilePathName)
lenV = len(signalV)

noiseFilePathName =  'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2_16K' +'.wav'  
rateN, signalN = scipy.io.wavfile.read(noiseFilePathName)
lenN = len(signalN)

signalN = np.append(signalN, signalN)
signalN = np.append(signalN, signalN)
signalN = np.append(signalN, signalN)
signalN = signalN[0:1500000]
lenN = len(signalN)

combined = signalN[0:lenV]/6 + signalV[0:lenV]/1.1
combinedFile = voiceFileName.replace('.wav', '.noise.wav')
combinedFilePathName = voicePathName + combinedFile
#
if os.path.isfile(combinedFilePathName):
    os.remove(combinedFilePathName)
#
WIDTH=2
wavef = wave.open(combinedFilePathName,'w')
wavef.setnchannels(1) # mono
wavef.setsampwidth(WIDTH) # two bytes per sound bit
wavef.setframerate(16000)
#
fmt = 'h'
#
for i in range(1, len(combined)):
    value = int(combined[i])
    data = struct.pack(fmt, value)
    wavef.writeframesraw(data)
#
data = struct.pack(fmt, 0)
wavef.writeframes(data)
wavef.close()




freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds


# load the background white noise file
noiseFilePathName = 'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2_16K' +'.wav'
#
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(noiseFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
spect_noise = torch.as_tensor(spectrogram, dtype=torch.float)
spect_noise /= 25  # The noise was scaled down in amplitude when it was mixed with the pure voice.  It needs to be scaled down by the same scale.
(_, _, frameCnt_noise) = np.shape(spect_noise)


# load a sample audio file
pathName = "C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\volice\\"
audioFile = "removenoise"
#
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


# load a previously saved spectrogram
savedWavFileName = 'cleaned_spect2'
savedWavFilePathName = pathName + savedWavFileName
spect_mixed = torch.load(savedWavFilePathName)
np.shape(spect_mixed)
#



# load a previously saved spectrogram
savedWavFileName = 'cleaned_spect3' + '.idx.1'
savedWavFilePathName = pathName + savedWavFileName
spect_cleaned1 = torch.load(savedWavFilePathName)
np.shape(spect_cleaned1)
#
savedWavFileName = 'cleaned_spect3' + '.idx.2'
savedWavFilePathName = pathName + savedWavFileName
spect_cleaned2 = torch.load(savedWavFilePathName)
np.shape(spect_cleaned2)
#
spect_cleaned = torch.cat((spect_cleaned1[:,:,0:485], spect_cleaned2[:,:,15:547]), dim=2)
np.shape(spect_cleaned)
torch.save(spect_cleaned, pathName + "cleaned_spect3" )


frameCnt=800
frameSkip=770
idx = 2
frameS = (idx-1) * frameSkip
frameE = frameS + frameCnt
spect_mixed_ = spect_mixed[:,:, frameS:frameE]
phasegram_ = phasegram[:,:, frameS:frameE]


frameCnt=500
spect_mixed_ = spect_mixed[:,:, 0:500]
phasegram_ = phasegram[:,:, 0:500]
#
frameCnt=547
spect_mixed_ = spect_mixed[:,:, 470:1017]
phasegram_ = phasegram[:,:, 470:1017]

frameSeq = None
frameSeq = voice_enhancement(classifier, softmask, spect_mixed_, frameCnt, MARGIN)

# concatenate all frames into a pytorch tensor
spect_cleaned = torch.cat(frameSeq, dim=2)
torch.save(spect_cleaned, pathName + "cleaned_spect3" + ".idx." + str(idx))

# generate a wave file
generatedWavFileName = 'removenoise.cleaned3.wav'
generatedWavFilePathName = pathName + generatedWavFileName
FourierTransform.playSound(frame_size, frame_stride, spect_cleaned, phasegram, frqcyBnd, sample_rate, 0, 10.18, generatedWavFilePathName)

