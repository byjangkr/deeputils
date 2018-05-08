#!/usr/bin/env python
# webrtcvad 
#   Copyright (C) 2016 John Wiseman
#   author : John Wiseman jjwiseman@gmail.com
#   license : MIT
# vadwav
#   modified by byjang 2018.05.08
#   author : Byeong-Yong Jang darkbulls44@gmail.com

import scipy
import sys
from scipy.io import wavfile
import webrtcvad
import collections
import matplotlib.pyplot as plt
import numpy

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate, padding=False):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    if padding:
        pad_size = (n-len(audio)%n)
        pad_audio = numpy.pad(audio,(0,pad_size+1),'edge')
        audio = pad_audio
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def reverse_min_sil_index(vad_frame_index_, min_sil_frames):
    vadinx = vad_frame_index_[:]
    _vad_frame_index = numpy.ones(len(vadinx))
    mval = min_sil_frames

    bufinx = []
    for i in xrange(1,len(vadinx)):
        if (vadinx[i-1] == 1) & (vadinx[i] == 0):
            bufinx.append(i)
        elif (vadinx[i-1] == 0) & (vadinx[i] == 0):
            bufinx.append(i)
        elif (vadinx[i-1] == 0) & (vadinx[i] == 1):
            print len(bufinx)
            if(len(bufinx) > mval):
                _vad_frame_index[bufinx] = 0
            bufinx = []

    return _vad_frame_index


def decision_vad_index(wav_path,vad_aggressive,vad_frame_size,min_sil_frames,log_level=0,fignum=1):
    # wav read for VAD
    vad_frame_index = []
    sample_rate, wav_data = scipy.io.wavfile.read(wav_path)

    if len(wav_data.shape) > 1:
        wav_data = (wav_data[:,0] + wav_data[:,1])/2

    vad = webrtcvad.Vad()
    vad.set_mode(vad_aggressive)
    frames = frame_generator(vad_frame_size, wav_data, sample_rate, padding=True)
    frames = list(frames)
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        vad_frame_index.append(1 if is_speech else 0)

    if log_level > 0:
        print "LOG: reverse index of silence less than %d frames" % (min_sil_frames)

    vad_frame_index_rev = reverse_min_sil_index(vad_frame_index, min_sil_frames)

    if log_level > 1:
        # check min silence frame
        # if the scale is too large, the graph is hard to see

        if(len(vad_frame_index) > 2000):
            endi = 2000
        else:
            endi = len(vad_frame_index)
        fr = range(0,endi)
        fig = plt.figure(fignum)
        fignum += 1
        fig.suptitle('Check median filter of vad result', fontsize=14)
        plt.subplot(211)
        plt.plot(fr,vad_frame_index[0:endi])
        plt.axis([fr[0],fr[-1],-0.5,1.5])
        plt.title('Origianl data')

        plt.subplot(212)
        plt.plot(fr,vad_frame_index_rev[0:endi])
        plt.axis([fr[0],fr[-1],-0.5,1.5])
        plt.title('After median filtering')




    if log_level > 0:
        print 'LOG: extend of vad result from frame unit to sample unit'
    # repeat matrix [window_sample * frame_size]
    vad_frame_index_rev_rep = numpy.matlib.repmat(vad_frame_index_rev,len(frame.bytes),1)
    [row,col] = vad_frame_index_rev_rep.shape
    # 1 frame -> N window_sample
    out_vad_index = numpy.reshape(numpy.transpose(vad_frame_index_rev_rep),[row*col,])

    # remove padded data
    if row*col > len(wav_data):
        if log_level > 0:
            print 'LOG: remove padded data'
        pad_size = row*col - len(wav_data)
        out_vad_index = out_vad_index[0:len(out_vad_index)-pad_size]

    return out_vad_index, wav_data

def is_voice_frame(vad_index,proportion=0.1):
    fsize = len(vad_index)
    sdata = numpy.sum(vad_index)
    if sdata < fsize*proportion:
        return False
    else:
        return True

def plotvad(vad_index,wav_data,fignum):
    # VAD index padding with length of wav data
    if len(vad_index) != len(wav_data) :
        pad_size = len(wav_data) - len(vad_index)
        vad_index.extend(scipy.zeros(pad_size))

    # normalizing
    no_data = wav_data/float(2**15-1) # range -1 ~ 1
    vad_index = numpy.array(vad_index)*max(no_data)

    # plot wav and VAD index
    plt.figure(fignum)
    plt.title('Signal wave and vad index...')
    plt.plot(no_data)
    plt.plot(vad_index,color="red")
    plt.show()


def main():
    loglev = 3
    print "VAD example code..."
    wav_file='../sample_data/speech_sample.wav'
    wav_file1 = '../sample_data/drama_sample_stereo.wav'
    vad_index, wav_data = decision_vad_index(wav_file,vad_aggressive=3,vad_frame_size=10,min_sil_frames=80,log_level=loglev)
    plotvad(vad_index,wav_data,2)

    if loglev > 2:
        plt.show()

if __name__=="__main__":
    main()