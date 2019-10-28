#Setup: importing packages
import numpy as np
import pandas as pd

import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

import math        
import pyaudio     #sudo apt-get install python-pyaudio

import os
import wave
import struct
import seaborn as sb 

#from numba import jit



class FourierTransform(object):  

    """
    https://en.wikipedia.org/wiki/Voice_frequency
    
    A voice frequency (VF) or voice band is one of the frequencies, within part of the audio range, that is being used for the transmission of speech.
    In telephony, the usable voice frequency band ranges from approximately 300 Hz to 3400 Hz. 
    It is for this reason that the ultra low frequency band of the electromagnetic spectrum between 300 and 3000 Hz is also referred to as voice frequency, 
    being the electromagnetic energy that represents acoustic energy at baseband.
    
    Per the Nyquist–Shannon sampling theorem, the sampling frequency (8 kHz) must be at least twice the highest component of the voice frequency 
    via appropriate filtering prior to sampling at discrete times (4 kHz) for effective reconstruction of the voice signal.
    
    Useful links regarding Discrete Fourier Transform applications on sound signal processing.
    https://www.youtube.com/watch?v=g1_wcbGUcDY
    
    """
    FreqBandLow  = 0.0
    FreqBandHigh = 20000.0

    @staticmethod
    def FFT(wavFilePathName, melFreqBandCnt, frame_size, frame_stride, duration=0, emphasize=True):
        '''
        wavFilePathName: full file path name of the wav file
        duration: number of seconds to sample from the wav file
                  if left unspecified (duration=0), the function would transform and return the entire wav file.
        '''
        # load the audio file
        sample_rate, signal = scipy.io.wavfile.read(wavFilePathName)
        sample_cnt = len(signal)
        if signal.ndim == 1:
            channel = 1
        else:
            signal = np.transpose(signal)
            (channel, _) = np.shape(signal)
        #
        if duration>0:
            # only transform the specified duration (in seconds)
            signal = signal[0:int(duration * sample_rate)]
        #
        spectrogram = [None  for _ in range(0, channel)]
        phasegram   = [None  for _ in range(0, channel)] 
        frqcyBnd    = [None  for _ in range(0, channel)]
        fft         = [None  for _ in range(0, channel)]
        #
        if channel==1:
            spectrogram[0], phasegram[0], frqcyBnd[0], fft[0] = FourierTransform.soundChannel(sample_rate, signal, frame_size, frame_stride, melFreqBandCnt, emphasize)
        else:
            for i in range(0, channel):
                spectrogram[i], phasegram[i], frqcyBnd[i], fft[1]= FourierTransform.soundChannel(sample_rate, signal[i], frame_size, frame_stride, melFreqBandCnt, emphasize)
        #
        spectrogram = np.array(spectrogram)
        phasegram   = np.array(phasegram)
        #
        #TEST GRAPH:   Plot each individual frame on the same graph
        (_, frqBndCnt, frameCnt) = np.shape(spectrogram)
        x = frqcyBnd[0] 
        #x = np.linspace(1, frqcyBnd[0][frqBndCnt-1], frqBndCnt)  
        '''
        for c in range(0, channel):
            for f in range(0, 100):
                # randomly select 100 frames to show in a plot
                n = np.random.randint(0, frameCnt)
                plt.plot(x, np.abs(spectrogram[c, :, n]))
            print(wavFilePathName, "  sound channel: ", c)
            plt.show()
            print()
        #
        '''
        return channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt, fft



    @staticmethod
    def soundChannel(sample_rate, signal, frame_size, frame_stride, melFreqBandCnt, emphasize=True):
        '''
        Perform a fast Fourier transfrom of the given sound bit file (signal) into freqBandCnt frequency bands.
        
        sample_rate:  number of sound sampling per second
        signal:  a single dimensional array in which each element represents a sound bit sampling in a WAV file
        FRAMING: Splitting audio file into very small and overlapping bits
            frame_size= length of frame(in second); 
            frame_stride= length of gap between frames(in second)
        melFreqBandCnt:  number of Mel frequency bands; by using the Mel frequency bands, the phase information for each frequency would be lost
        '''
        #
        # PRE-EMPHASIS: Amplifying higher frequencies
        if emphasize:
            pre_emphasis = 0.97 #Generally .95 or .97
            emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1]) #Applies pre-emphasis filter
        else:
            emphasized_signal = signal
        #
        # frame_length= number of samples within a single frame; frame_step= number of samples between each frame
        # frame_length/2 represents the maximum number of frequencies the Fourier transform can possibility detect
        frame_length, frame_step = int(frame_size * sample_rate), int(frame_stride * sample_rate)
        #
        num_frames = int(np.ceil( (len(emphasized_signal)-frame_length) / frame_step))
        #
        # Create 2D matrix, each row represents a frame, each column represetns a sound sample
        frames = np.zeros((num_frames, frame_length))
        #
        counter=0
        for f in range(0, len(emphasized_signal)-frame_length, frame_step):
            frames[counter] = emphasized_signal[f:f+frame_length]
            counter += 1
        #
        #
        # Apply Hamming window to each frame.
        frames *= np.hamming(frame_length) 
        #
        maxFreq = sample_rate/2
        maxFreqFFT = frame_length/2
        #
        # A spectrogram that contains no information about the exact or even approximate phase of the signal is not possible to reverse the process and generate a copy of the original signal.
        # The result for FFT is a set of complex numbers represent both the phase and magnitude of the signal
        #
        # FOURIER TRANSFORM
        # amplitude: list in which each element represents the amplitute of a frequency band
        # the maximum frequency the Fourier transform can detect is half of the frame_length
        #
        fft = scipy.fftpack.fft(frames)
        fftResult = fft[:,:int(frame_length/2)]
        amplitude = np.abs(fftResult)
        phase     = np.angle(fftResult)
        #
        # The highest frequency detected by the fft is frame_length/2, while the actual maximum frequency is sample_rate.
        # Hence it is necessary to adjust the fft output values to the actual hertz values.
        scale = (maxFreq) / (maxFreqFFT)
        #
        print("sample rate = ", sample_rate)
        print("maximum frequency of the sound recording = ", maxFreq)
        print("maximum frequency by FFT = ", maxFreqFFT)
        print("frequency scaling factor from fft to actual hertz = ", scale)
        #
        # calculate the correspoinding fft band range given the frequency lower and upper bond
        fftLow  = int(FourierTransform.FreqBandLow  / scale)
        fftHigh = min(int(FourierTransform.FreqBandHigh / scale), int(maxFreqFFT)-1)
        print("FFT frequency range for voice band = [", fftLow, ", ", fftHigh, "]")
        #
        frqcy = None
        spectrogram = None
        phasegram = None
        #
        if melFreqBandCnt>0:  # use Mel frequency bands
            #
            # Defines frequency channels such that the frequency range of each is equal in terms of mel-scale 
            high_mel = 2595 * np.log10(1+(FourierTransform.FreqBandHigh)/700)  # Calculates the highest possible mel value given the input requency range
            mels = np.linspace(0, high_mel, melFreqBandCnt+2) 
            hzs = 700 * (np.power(10, (mels/2596))-1) 
            #
            # create a array to store the frequency in hertz
            frqcy = np.zeros((len(hzs)-2))
            #
            spectrogram = np.zeros((melFreqBandCnt, num_frames))
            phasegram = None  # phase information is lost when using Mel frequency bands
            #
            # Loops through each frequency channel
            for h in range(1, len(hzs)-1):
                # Retrieves three points on current frequency channel
                tleft = hzs[h-1]
                tright = hzs[h+1]
                tmid = (tright+tleft)/2
                frqcy[h-1] = tmid
                #
                # Calculates slope of current frequency channel
                slope = 1/(tmid-tleft)
                #
                #print("left range: [", int(tleft/scale), ", ", int(tmid/scale), "]", "   right range: [", int(tmid/scale), ", ", int(tright/scale), "]")
                #
                # Loops through each frame,   within frequency channel, stores it in spectrogram
                for t in range(0, len(amplitude)-1):
                    amp = 0
                    #
                    vleft  = int(tleft/scale) if (tleft%scale)==0 else int(tleft/scale)+1  # small integer greater than or equal to tleft/scale
                    vmid_l = int(math.ceil(tmid/scale)-1)                                  # largest integer lest than timd/scale
                    vmid_r = int(tmid/scale) if (tmid%scale)==0 else int(tmid/scale)+1     # small integer greater than or equal to tmid/scale
                    vright = int(math.ceil(tright/scale)-1)                                # largest integer lest than tright/scale
                    #
                    # Loops through each amplitude in amplitude within the first half of the frequency channel
                    for v in range(vleft, vmid_l+1):
                        amp += (v*scale-tleft) * slope * amplitude[t][v]
                    #
                    # Loops through each amplitude in amplitude within the second half of the frequency channel
                    for v in range(vmid_r, vright+1):    
                        amp += (tright-v*scale) * slope * amplitude[t][v]
                    #
                    spectrogram[h-1][t] = amp
                    #
                #
            #
        #
        else:  # use original frequency band from Fourier transform; the frequency phase information is preserved
            #
            # create a array to store the frequency in hertz
            fftFreqBandCnt = fftHigh - fftLow + 1
            frqcy = np.zeros(fftFreqBandCnt)
            phasegram = phase
            #
            spectrogram = np.zeros((fftFreqBandCnt, num_frames)) 
            phasegram = np.zeros((fftFreqBandCnt, num_frames))
            #
            for h in range(fftLow, fftHigh+1):
                frqIdx = h-fftLow
                frqcy[frqIdx] = h*scale
                #
                for t in range(0, len(amplitude)):
                    spectrogram[frqIdx][t] = amplitude[t][h]
                    phasegram[frqIdx][t] = phase[t][h]
                #
            #
        #
        # NORMALIZING: rescale the strenght of each frequency band to the range of [0, 1]
        spectrogramN = np.copy(spectrogram)
        #
        mx = np.max(spectrogramN)
        mn = np.min(spectrogramN)
        spectrogramN = spectrogramN / (mx-mn)
        #    
        mn = np.min(spectrogramN)
        spectrogramN = spectrogramN - mn
        #
        return spectrogramN, phasegram, frqcy, fft


    @staticmethod
    def generateSoundWave(durationF, soundBitCnt, soundBitPerStride, soundBitPerFrame, spectrogram, phasegram, frqcy, channel, sample_rate):
        """
        durationF:   nunber of sound frames to generate
        soundBitCnt: total number of sound bits to generate
        soundBitPerStride: number of sound bits per stride
        soundBitPerFrame:  number of sound bits per frame
        spectrogram:
        phasegram:
        channel:
        sample_rate:
        """
        #
        PI = math.pi
        #
        _, frqcyBandCnt, _ = np.shape(spectrogram)
        #
        # the array to which the sound wave is generated
        buffer = np.array([0.0  for _ in range(0, soundBitCnt)])
        #
        # loop through each frame within the given duration
        for fIdx in range(0, durationF):
            s =  fIdx * soundBitPerStride # the starting point of the sound bit for the given fram fIdx
            #
            # loop through each frequency band to generate the corrensponding audio wave
            for hIdx in range(0, frqcyBandCnt):
                amp = spectrogram[channel, hIdx, fIdx]   # the amplitute for a frequency
                hz = frqcy[channel][hIdx]             # the frequency in Hertz
                #
                for sIdx in range(s, s+soundBitPerFrame):                    
                    #
                    if phasegram is None:
                        # Generate sound waves at the given time point and ignore phase info (assume that all frequencies start with phase 0 at time 0)
                        time = float(sIdx) / float(sample_rate)   # the actual time point in second
                        frqWave = amp * math.sin(2*PI*time*hz)  
                    else:
                        # Generate sound waves at the given time point and incorporating phase info
                        time = float(sIdx-s) / float(sample_rate)  # the time point relative to the frame (ie time=0 at the beginning of a frame)
                        angle = phasegram[  channel, hIdx, fIdx]
                        frqWave = amp * math.sin(2*PI*time*hz+angle)  
                    #
                    buffer[sIdx] += frqWave     # accumulate all sound wave the the given time point
                #
            #
            print("generateSoundWave, processed frame: ", fIdx, ", out of ", durationF)
        #
        # NOTE:  The following code use ifft() function to inverse a FFT result back to the original signal
        #x = np.arange(5)
        #result = scipy.fftpack.fft(x)
        #x_ = scipy.fftpack.ifft(result)
        #np.allclose(x_, x, atol=1e-15)  # within numerical accuracy.
        #
        return buffer


    @staticmethod
    def playSound(frame_size, frame_stride, spectrogram, phasegram, frqcy, sample_rate, channel, duration, generatedWavFilePathName):  
        '''
        Given a spectrogram, and an optional phasegram, generate a WAV file representing the original audio
        
        spectrogram: the amplitude matrix for frequency bands and time (as a result of Fourier transform)
        phasegram:   phase information for each frequency
        frqcy:       the actual frequencies in Hertz for each frequency band
        duration:    number of seconds of sound to generate
        '''
        #
        soundBitCnt = int(duration * sample_rate)
        soundBitPerFrame  = int(sample_rate * frame_size)
        soundBitPerStride = int(sample_rate * frame_stride)
        #
        (_, _ , frameCnt) = np.shape(spectrogram) 
        #
        durationF = int((duration-frame_size)/frame_stride + 1.0)
        #
        buffer = FourierTransform.generateSoundWave(durationF, soundBitCnt, soundBitPerStride, soundBitPerFrame, spectrogram, phasegram, frqcy, channel, sample_rate)
        #
        # normalize the buffer value to [0, 1]
        bufferN = np.copy(buffer)
        #
        mx = np.max(bufferN)
        mn = np.min(bufferN)
        bufferN = bufferN / (mx-mn)
        #    
        mn = np.min(bufferN)
        bufferN = bufferN - mn    
        #
        # normalize the buffer value to [-1, 1]
        bufferN = bufferN*2 - 1
        #
        # generate a wave file
        # the WIDTH can be either 2 or 4
        WIDTH =  2
        K = 2**(WIDTH*8) / 2 - 1
        #
        wavef = wave.open(generatedWavFilePathName,'w')
        wavef.setnchannels(1) # mono
        wavef.setsampwidth(WIDTH) # two bytes per sound bit
        wavef.setframerate(sample_rate)
        #
        fmt = 'h' if WIDTH==2 else 'l'
        #
        for i in range(1, len(bufferN)):
            value = int(K * bufferN[i])
            data = struct.pack(fmt, value)
            wavef.writeframesraw(data)
        #
        data = struct.pack(fmt, 0)
        wavef.writeframes(data)
        wavef.close()
        #
        #
        # generate the wave file
        """
        PyAudio = pyaudio.PyAudio     #initialize pyaudio
        #
        waveByte = ''
        #
        for i in range(0, len(bufferN)):
            waveByte += chr(int(bufferN[i]*127 + 128))
        #
        p = PyAudio()
        stream = p.open(format = p.get_format_from_width(1), 
                        channels = 1, 
                        rate = sample_rate, 
                        output = True)
        stream.write(waveByte)
        stream.stop_stream()
        stream.close()
        p.terminate()
        #
        """
        return



###################################################################################################################


class soundWAVUtil(object):
    
    @staticmethod
    def wavFileHeatmap(sig, sample_rate, melFreqBandCnt, frame_size, frame_stride, frqcyRange=None, frameRange=None):
        '''
        sig: signal from wav file
        sample_rate: sample_rate of wav file
        melFreqBandCnt: number of Mel frequency bands, use 0 to keep original frequencies without Mel transform
        frame_size: length of each frame (in seconds) for Fourier Transform
        frame_stride: length between each frame (in seconds) 
        frqcyRange: could be either None of a list of two elements [frqBndStrat, frqBndEnd]
        
        returns spectrogram values
        '''
        #
        spectrogram, phasegram, frqcy, fft = FourierTransform.soundChannel(sample_rate, sig, frame_size, frame_stride, melFreqBandCnt, emphasize=True)
        if frqcyRange == None: frqcyRange=[0,len(spectrogram)]
        if frameRange == None: frameRange=[0,len(spectrogram[0])]
        sb.heatmap(spectrogram[frqcyRange[0]:frqcyRange[1]][frameRange[0]:frameRange[1]]).invert_yaxis()
        return spectrogram
    
    '''
    To test wavFileHeatmap()
    
    wavFilePathName = ['/Users/newacc/Testing/SX327.WAV.noise.wav','SX327.WAV.wav','Whitenoise2_16K.wav']
    sample_rate, sig = scipy.io.wavfile.read(wavFilePathName[2])
    
    melFreqBandCnt = 0
    frame_size = .03
    frame_stride = .01
    
    spectrogram(sig, sample_rate, melFreqBandCnt, frame_size, frame_stride)
    
    ''' 
    
    @staticmethod
    def spectrogramHeatmap(spectrogram, frqyScale=None, frqcyRange=None, frameRange=None, ampRange=None):
        if frqcyRange == None: frqcyRange=[0,len(spectrogram)]
        if frameRange == None: frameRange=[0,len(spectrogram[0])]
        if ampRange == None:
            sb.heatmap(spectrogram[frqcyRange[0]:frqcyRange[1]][frameRange[0]:frameRange[1]]).invert_yaxis()
        else:
            sb.heatmap(spectrogram[frqcyRange[0]:frqcyRange[1]][frameRange[0]:frameRange[1]], vmin=ampRange[0],vmax=ampRange[1]).invert_yaxis()
    
    
    
    def combineAudio(sourceFilePathNames, duration, mixedFilePathName, amplifyIdx=1, startIdx=0):
        '''
        Combines two samples of audio (wav files).
        
        sourceFilePathNames = list of two file paths to be combined
        duration = length of audio to be combined in seconds
        mixedFilePathName = name of resulting combined file
        amplifyIdx = # of times louder the first file will be compared to second in combined audio
        startIdx = index of sample to start combining from
        '''
        # 
        sample_rate, signal =[[None for _ in (0,len(sourceFilePathNames))] for _ in (0,2)]
        #
        sample_rate[0], signal[0] = scipy.io.wavfile.read(sourceFilePathNames[0])
        sample_rate[1], signal[1] = scipy.io.wavfile.read(sourceFilePathNames[1])
        #
        if(sample_rate[0] != sample_rate[1]):
            print('Error: Sample rates of files do not match')
            return
        #
        if(len(signal[0]) < (duration+startIdx)*sample_rate[0] or len(signal[1]) < (duration+startIdx)*sample_rate[0]):
            print('Error: Audio sample(s) are too short for specified duration')
            return
        #
        signal[0] = signal[0]*amplifyIdx 
        #
        endIdx = startIdx + int(duration * sample_rate[0])
        result = signal[0][startIdx:endIdx] + signal[1][startIdx:endIdx]
        #
        WIDTH =  2 
        #
        wavef = wave.open(mixedFilePathName,'w')
        wavef.setnchannels(1) # mono
        wavef.setsampwidth(WIDTH) # two bytes per sound bit
        wavef.setframerate(sample_rate[0])
        #
        fmt = 'h' if WIDTH==2 else 'l'
        #
        for i in range(1, len(result)):
            value = int(result[i])
            data = struct.pack(fmt, value)
            wavef.writeframesraw(data)
        #
        data = struct.pack(fmt, 0)
        wavef.writeframes(data)
        wavef.close()
        #
        return result
        
        
    '''
    To test combineAudio():

    folderPath = '/Users/newacc/Testing/'
    #fileNames = ['financial_commentary1.wav', 'rain2.wav', 'wind01.wav', 'crowdhomerunapplause.wav', 'Corvette_pass.wav', 'lawnmower.wav', 'applause7.wav', 'police_sirens.wav', 'white_noise.wav', 'steps.wav', 'techno_drum.wav']
    fileNames = ['/Users/newacc/Testing/SX327.WAV.noise.wav','SX327.WAV.wav','Whitenoise2_16K.wav']
    filePaths = [folderPath + fileNames[1],folderPath + fileNames[2]]
    
    duration = 3
    amplifyIdx=4
    startIdx=0
    mixedFilePathName = str(fileNames[0])[:-4] + '_' + str(fileNames[1])[:-4] + '(amplifyIdx=' + str(amplifyIdx) + ').wav'
    
    combineAudio(filePaths, duration, mixedFilePathName, amplifyIdx, startIdx)
    '''

###################################################################################
###################################################################################

    

"""
filePath = "C:\\Data\\WAV\\Environment\\"
fileNames = ["crowd.wav", "highway.wav", "rain.wav", "violin.wav", "piano.wav", "beach.wav", "birdChirping.wav", "tekTalk.wav"]
wavFilePathName = filePath + fileNames[3]

wavFilePathName = filePath + "audiocheck.net_sin_200Hz_-3dBFS_48k.wav"
wavFilePathName = filePath + "audiocheck.net_sin_500Hz_-3dBFS_48k.wav"
wavFilePathName = filePath + "audiocheck.net_sin_1000Hz_-3dBFS_8k.wav"
wavFilePathName = filePath + "TedTalk.wav"
wavFilePathName = filePath + "financial_commentary1.wav"

freqBandCnt=400
(frame_size, frame_stride) = (0.030, 0.010)
"""

filePath = 'C:\\Data\\WAV\\Environment\\'
fileNames = ['speechlong.wav', 'water-rain1.wav', 'wind01.wav', 'Corvette_pass.wav', 'lawnmower.wav', 'applause7.wav', 'police_sirens.wav', 'white_noise.wav', 'steps.wav', 'techno_drum.wav']

len(fileNames)
wavFilePathName = filePath + fileNames[0]

melFreqBandCnt=0
melFreqBandCnt=300
(frame_size, frame_stride) = (0.030, 0.010)

for c in range(0, len(fileNames)):
    wavFilePathName = filePath + fileNames[c]
    channel, spectrogram, phasegram, frqcy, sample_rate = FourierTransform.FFT(wavFilePathName, melFreqBandCnt, frame_size, frame_stride, duration=0, emphasize=False)



freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds

# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR1\\MTRR0\\"
audioFile = "SI918.WAV"
#
# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TEST\\DR6\\MRJR0\\"
audioFile = "SI2313.WAV"
#

wavfileName = audioFile + ".wav" 
wavFilePathName = pathName + wavfileName

channel, spectrogram, phasegram, frqcy, sample_rate, sample_cnt, fft = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)


channel=0
duration=3.5
generatedWavFilePathName = "C:\\Data\\WAV\\cleaned3.wav"

FourierTransform.playSound(frame_size, frame_stride, spectrogram, phasegram, frqcy, sample_rate, channel, duration, generatedWavFilePathName)



spect_mixed_np = np.array(spect_mixed)
spect_cleaned2_np = np.array(spect_cleaned2.detach())
spect_cleaned3_np = np.array(spect_cleaned3.detach())
spect_cleaned5_np = np.array(spect_cleaned5.detach())

FourierTransform.playSound(frame_size, frame_stride, spect_mixed_np, phasegram, frqcy, sample_rate, channel, duration, generatedWavFilePathName)
FourierTransform.playSound(frame_size, frame_stride, spect_cleaned2_np, phasegram, frqcy, sample_rate, channel, duration, generatedWavFilePathName)
FourierTransform.playSound(frame_size, frame_stride, spect_cleaned3_np, phasegram, frqcy, sample_rate, channel, duration, generatedWavFilePathName)

FourierTransform.playSound(frame_size, frame_stride, spectrogram, None,      frqcy, sample_rate, channel, duration, generatedWavFilePathName)


wavFilePathName = filePath + "audiocheck.net_sin_500Hz_-3dBFS_48k.wav"


###################################################################################
# testing the fft function
#
from scipy.fftpack import fft
import matplotlib.pyplot as plt
 
# Number of sample points
N = 400
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)

y1 = 2*np.sin(10.0 * 2.0*np.pi*x)
plt.plot(x, 2.0/N * np.abs(y1))

y2 = np.sin(20.0 * 2.0*np.pi*x)
plt.plot(x, 2.0/N * np.abs(y2))

y = y1 +y2
plt.plot(x, 2.0/N * np.abs(y))

yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()


###################################################################################



def sound_generator(sample_rate, freq):
    #
    PyAudio = pyaudio.PyAudio     #initialize pyaudio
    #
    amp = 0.5
    twoPi = 2*math.pi;
    #
    buffer = [None  for _ in range(0, sample_rate)]
    
    for i in range(0, len(buffer)):
      t = i / sample_rate;
      buffer[i] = amp * math.sin(twoPi*freq*t)
    #
    wave = ''
    #
    for i in range(0, len(buffer)):
        wave += chr(int(buffer[i]*127+128))
    #
    p = PyAudio()
    stream = p.open(format = p.get_format_from_width(1), 
                    channels = 1, 
                    rate = sample_rate, 
                    output = True)
    stream.write(wave)
    stream.stop_stream()
    stream.close()
    p.terminate()




sample_rate = 48000
freq = 500

sound_generator(sample_rate, freq)
        
        
###################################################################################

##  http://blog.acipo.com/wave-generation-in-python/

import wave, struct, math

sampleRate = 44100.0 # hertz
duration = 1.0       # seconds
frequency = 200.0    # hertz

filePath = "C:\\Data\\WAV\\Environment\\"
fileNames = ["test.wav"]
wavFilePathName = filePath + fileNames[0]

wavef = wave.open(wavFilePathName,'w')
wavef.setnchannels(1) # mono
wavef.setsampwidth(2) 
wavef.setframerate(sampleRate)

values = []

for i in range(1, int(duration * sampleRate)):
    value = int(32767.0*math.cos(frequency*math.pi*float(i)/float(sampleRate)))
    data = struct.pack('<h', value)
    values.append(value)
    #wavef.writeframesraw( data )

data = struct.pack('<h', 0)
wavef.writeframes(data)
wavef.close()


###################################################################################

def combineAudio(filePaths, duration, fileName, amplifyIdx=1, startIdx=0):
    '''
    Combines two samples of audio (wav files).
    
    filePaths = list of two file paths to be combined
    duration = length of audio to be combined in seconds
    fileName = name of resulting combined file
    amplifyIdx = # of times louder the first file will be compared to second in combined audio
    startIdx = index of sample to start combining from
    '''
    
    #folderPath = 'C:\\Data\\WAV\\Environment\\'
    #fileNames = ['audiobook.wav', 'water-rain1.wav', 'wind01.wav', 'crowdhomerunapplause.wav', 'Corvette_pass.wav', 'lawnmower.wav', 'applause7.wav', 'police_sirens.wav', 'white_noise.wav', 'steps.wav', 'techno_drum.wav']
    #filePaths = [folderPath + fileNames[0], folderPath + fileNames[10]] 
    
    sample_rate, signal =[[None for _ in (0,len(filePaths))] for _ in (0,2)]
    
    sample_rate[0], signal[0] = scipy.io.wavfile.read(filePaths[0])
    sample_rate[1], signal[1] = scipy.io.wavfile.read(filePaths[1])
    

    sig = [None for _ in (0,len(signal))]
    # only transform the specified duration (in seconds)
    for s in range(0,len(signal)):
        sig[s] = signal[s][startIdx:startIdx+int(duration * sample_rate[s])]
        
    s0max=max(max(sig[0]),abs(min(sig[0])))
    s1max=max(max(sig[1]),abs(min(sig[1])))    
    
    adjustedBg=sig[1]*(s0max/(s1max*amplifyIdx))
        
    result = (sig[0]+adjustedBg)/2
    
    
    generatedWavFilePathName = 'C:\\Data\\WAV\\Environment\\' + fileName +'.wav'
    
    WIDTH =  2 
    K = 2**(WIDTH*8) / 2 - 1
    #
    wavef = wave.open(generatedWavFilePathName,'w')
    wavef.setnchannels(1) # mono
    wavef.setsampwidth(WIDTH) # two bytes per sound bit
    wavef.setframerate(sample_rate[0])
    #
    fmt = 'h' if WIDTH==2 else 'l'
    #
    for i in range(1, len(result)):
        value = int(result[i])
        data = struct.pack(fmt, value)
        wavef.writeframesraw(data)
    #
    data = struct.pack(fmt, 0)
    wavef.writeframes(data)
    wavef.close()
    #
 
def combineAudio(filePathName1, filePathName2, duration, amplifyIdx=1, startIdx=0):     
    return None
 

def combineAudio(filePathName1, filePathName2, duration, amplifyIdx=1, startIdx=0):     
    return None



def generateNoisyAudio(dataframe, TIMITRootPath, dialects, noiseFilePathName):
    #
    TIMITRootPath = TIMITRootPath
    TIMITDataPath = TIMITRootPath + "Data\\"
    dialects  = dialects
    #
    dataframePathName = TIMITRootPath + dataframe
    dataSet = pd.read_pickle(dataframePathName)
    #
    trainSub = (dataSet.dialect_region.isin(dialects))
    dataSet = dataSet[trainSub]
    dataSet = dataSet.reset_index(drop=True)
    fileCnt = len(dataSet)
    #
    sample_rate1, signal1 = scipy.io.wavfile.read(noiseFilePathName)
    len1 = len(signal1)
    #
    WIDTH =  2 
    K = 2**(WIDTH*8) / 2 - 1
    #
    for i in range(0, fileCnt):
        #
        fileName = dataSet.loc[i].wav.replace('\\\\', '\\')
        filePathName = TIMITDataPath + fileName
        #
        sample_rate0, signal0 = scipy.io.wavfile.read(filePathName)
        #
        len0 = len(signal0)
        if len1>=len0:
            combined = signal0 + signal1[0:len0]/60
            combinedFile = fileName.replace('.wav', '.noise.wav')
            combinedFilePathName = TIMITDataPath + combinedFile
            #
            if os.path.isfile(combinedFilePathName):
                os.remove(combinedFilePathName)
            #
            wavef = wave.open(combinedFilePathName,'w')
            wavef.setnchannels(1) # mono
            wavef.setsampwidth(WIDTH) # two bytes per sound bit
            wavef.setframerate(16000)
            #
            fmt = 'h' if WIDTH==2 else 'l'
            #
            for i in range(1, len(combined)):
                value = int(combined[i])
                data = struct.pack(fmt, value)
                wavef.writeframesraw(data)
            #
            data = struct.pack(fmt, 0)
            wavef.writeframes(data)
            wavef.close()
        #
    #
        
        
 

TIMITRootPath = "C:\\Data\\TIMIT\\"
noiseFilePathName = 'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2_16K' +'.wav'       
dialects=['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']
       
generateNoisyAudio("train_data.df", TIMITRootPath, dialects, noiseFilePathName)
generateNoisyAudio("test_data.df",  TIMITRootPath, dialects, noiseFilePathName)
    

---------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------



generatedWavFilePathName = 'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2_16K' +'.wav'
sourceFile = 'C:\\Data\\WAV\\Environment\\' + 'Whitenoise2' +'.wav'
sample_rate, signal = scipy.io.wavfile.read(sourceFile)
signal = np.array(signal, dtype=np.int32)

cnt = len(signal)
r = cnt%3
newCnt = int(cnt/3)
buffer = np.array([0.0  for _ in range(0, newCnt)])
i=0

for idx in range(0, newCnt):
    buffer[idx] = (signal[i]+signal[i+1]+signal[i+2])/3.0
    i +=3

WIDTH =  2 
K = 2**(WIDTH*8) / 2 - 1
#
wavef = wave.open(generatedWavFilePathName,'w')
wavef.setnchannels(1) # mono
wavef.setsampwidth(WIDTH) # two bytes per sound bit
wavef.setframerate(16000)
#
fmt = 'h' if WIDTH==2 else 'l'
#
for _ in range(0,2):
    for i in range(1, len(buffer)):
        value = int(buffer[i])
        data = struct.pack(fmt, value)
        wavef.writeframesraw(data)
#
data = struct.pack(fmt, 0)
wavef.writeframes(data)
wavef.close()

