#!/usr/bin/env python
# webrtcvad 
#   Copyright (C) 2016 John Wiseman
#   author : John Wiseman jjwiseman@gmail.com
# vadwav
#   modified by byjang 2018.05.08
#   author : Byeong-Yong Jang darkbulls44@gmail.com
# VAD-python
#   Reference : https://github.com/marsbroshok/VAD-python.git

import scipy
from scipy.io import wavfile
import webrtcvad
import matplotlib.pyplot as plt
import numpy
from VAD_python.vad import VoiceActivityDetector as envad
from statistical_vad import vad as stvad

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

def plot_min_filter(vad_frame_index, vad_frame_index_rev,fignum):
    # if the scale is too large, the graph is hard to see

    if (len(vad_frame_index) > 2000):
        endi = 2000
    else:
        endi = len(vad_frame_index)
    fr = range(0, endi)
    fig = plt.figure(fignum)
    fignum += 1
    fig.suptitle('Check min sil filter of vad result', fontsize=14)
    plt.subplot(211)
    plt.plot(fr, vad_frame_index[0:endi])
    plt.axis([fr[0], fr[-1], -0.5, 1.5])
    plt.title('Origianl data')

    plt.subplot(212)
    plt.plot(fr, vad_frame_index_rev[0:endi])
    plt.axis([fr[0], fr[-1], -0.5, 1.5])
    plt.title('After min sil filtering')


def reverse_min_sil_index(vad_frame_index_, min_sil_frames):
    vadinx = vad_frame_index_[:]
    _vad_frame_index = numpy.ones(len(vadinx))
    mval = min_sil_frames

    intro_margin = min_sil_frames
    _vad_frame_index[0:intro_margin] = vadinx[0:intro_margin]

    bufinx = []
    for i in xrange(intro_margin,len(vadinx)):
        if (vadinx[i-1] == 1) & (vadinx[i] == 0):
            bufinx.append(i)
        elif (vadinx[i-1] == 0) & (vadinx[i] == 0):
            bufinx.append(i)
        elif (vadinx[i-1] == 0) & (vadinx[i] == 1):
            if(len(bufinx) > mval):
                _vad_frame_index[bufinx] = 0
            bufinx = []

        if i == len(vadinx)-1 and len(bufinx) != 0:
            _vad_frame_index[bufinx] = 0
            bufinx = []

    return _vad_frame_index

def decision_vad_index_with_statis_model(wav_path,vad_frame_size_,vad_fft_size, vad_shift_size_,min_sil_frames,log_level=0,fignum=1):
    # wav read for VAD
    vad_frame_size = vad_frame_size_ * 0.001
    vad_shift_size = vad_shift_size_ * 0.001
    sample_rate, wav_data = scipy.io.wavfile.read(wav_path)

    if len(wav_data.shape) > 1:
        wav_data = (wav_data[:,0] + wav_data[:,1])/2

    vad_frame_index = stvad.VAD(wav_data,sample_rate,nFFT=vad_fft_size,
                                win_length=vad_frame_size,hop_length=vad_shift_size,theshold=0.6)

    if log_level > 0:
        print "LOG: reverse index of silence less than %d frames" % (min_sil_frames)

    vad_frame_index = numpy.squeeze(numpy.array(vad_frame_index),axis=1)
    vad_frame_index_rev = reverse_min_sil_index(vad_frame_index, min_sil_frames)

    if log_level > 1:
        # check min silence frame
        plot_min_filter(vad_frame_index,vad_frame_index_rev,fignum)


    if log_level > 0:
        print 'LOG: extend of vad result from frame unit to sample unit'
    # repeat matrix [window_sample * frame_size]
    repsize = int(vad_shift_size*sample_rate)
    vad_frame_index_rev_rep = numpy.matlib.repmat(vad_frame_index_rev,repsize,1)
    [row,col] = vad_frame_index_rev_rep.shape
    # 1 frame -> N window_sample
    out_vad_index = numpy.reshape(numpy.transpose(vad_frame_index_rev_rep),[row*col,])

    # add vad data as long as length of wav
    if row*col < len(wav_data):
        if log_level > 0:
            print 'LOG: remove padded data'
        pad_size = len(wav_data) - row*col
        if pad_size > (vad_frame_size*sample_rate):
            print 'ERROR(vad): the large pad_size (%d) than repsize (%d)' %(pad_size,(vad_frame_size*sample_rate))
            raise()
        out_vad_index = numpy.pad(out_vad_index,(0,pad_size),'edge')

    return out_vad_index, wav_data

def decision_vad_index_with_energy(wav_path,vad_frame_size,min_sil_frames_,log_level=0, fignum=1):
    win_size = vad_frame_size * 0.001
    win_shift = win_size/2
    min_sil_frames = min_sil_frames_

    v = envad(wav_path)
    v.plot_detected_speech_regions()
    vad_frame_index, sample_rate, wav_data = v.detect_speech()
    vad_frame_index_rev = reverse_min_sil_index(vad_frame_index[:,1], min_sil_frames)
    if log_level > 1:
        # check min silence frame
        plot_min_filter(vad_frame_index,vad_frame_index_rev,fignum)

    repsize = int(win_shift * sample_rate)
    if repsize != (vad_frame_index[1,0] - vad_frame_index[0,0]):
        print "ERROR(vad): wrong frame shift size\n"
        raise()

    vad_frame_index_rev_rep = numpy.matlib.repmat(vad_frame_index_rev, repsize, 1)
    [row,col] = vad_frame_index_rev_rep.shape
    # 1 frame -> N window_sample
    out_vad_index = numpy.reshape(numpy.transpose(vad_frame_index_rev_rep),[row*col,])

    # add vad data as long as length of wav
    if row*col < len(wav_data):
        if log_level > 0:
            print 'LOG: remove padded data'
        pad_size = len(wav_data) - row*col
        out_vad_index = numpy.pad(out_vad_index,(0,pad_size),'edge')

    return out_vad_index, wav_data


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
        plot_min_filter(vad_frame_index,vad_frame_index_rev,fignum)


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
        if pad_size > len(frame.bytes):
            print 'ERROR(vad): wrong padding size'
            raise()
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
    # plt.show()


def main():
    loglev = 3
    print "VAD example code..."
    # wav_file='../sample_data/speech_sample.wav'
    # wav_file1 = '../sample_data/drama_sample_stereo.wav'
    # wav_file = '../sample_data/vad/mask_00001_700K_1668_1670.wav'
    wav_file = '../sample_data/vad/mask_00001_700K_2251_2253_short.wav'

    vad_index, wav_data = decision_vad_index(wav_file,vad_aggressive=0,vad_frame_size=10,min_sil_frames=70,log_level=0)
    # vad_index2, wav_data2 = decision_vad_index_with_energy(wav_file,vad_frame_size=20,min_sil_frames_=80,log_level=loglev)
    # vad_index3, wav_data3 = decision_vad_index_with_statis_model(wav_file,vad_frame_size_=25,vad_shift_size_=10,vad_fft_size=512,
                                                                 # min_sil_frames=80,log_level=0)
    plotvad(vad_index,wav_data,1)
    # plotvad(vad_index3,wav_data3,2)

    if loglev > 2:
        plt.show()

if __name__=="__main__":
    main()