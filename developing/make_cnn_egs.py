"""
    Copyright 2018.4. Byeong-Yong Jang
    byjang@cbnu.ac.kr
    This code is for creating CNN input.
    Extract spectrogram and detect voice activity.
    If do you want check result and log, then change the log_level to 3.

    Usage
    -----
    python make_cnn_egs.py --sample-frequency=16000 --frame-length=25 \
                           --frame-shift=10 --fft-size=512 \
                           --vad-agressive=3 --vad-frame-size=10 \
                           --vad-medfilter-size=5 --target-label=speech
                           test.wav  test.bin

    Input
    -----
    wav-file : path of wave file
    out-file : path of output file (pickle format)
               out_data = (spec_data, vad_data, label)


    Options
    -------
    --sample-frequency (int) : sample rate of wav (Hz) [default: 16000]
    --frame-length (int) : frame size (ms) [default: 25 ms]
    --frame-shift (int) : frame shift size (ms) [default: 10 ms]
    --fft-size (int) : FFT window size (sample) [default: frame_size*sample_frequency]

    --vad-aggressive (int) : aggressive number for VAD (0~3 / least ~ most) [default: 3]
    --vad-frame-size (int) : frame size (only 10 or 20 or 30 ms) for VAD [default: 10]
    --vad-medfilter-size (int) : median filter size for VAD [default: 5 frame]

    --target-label (string) : target label [default: None]
    --save-length (int) : maximum length to save (sec) [default: wav length]
    --save-info (string) : save information file [optional]

"""

import pickle
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import scipy
import webrtcvad
from numpy import matlib
from scipy import signal
from scipy.io import wavfile

import extract_spec
import vadwav

### hyper parameters ###
log_level = 0
# 0 : default. do not print log message
# 1 : print log message, but do not plot results
# 2 : plot results

fignum = 1
### end of hyper parameter ###

def decision_vad(wav_path,vad_aggressive,vad_frame_size,vad_medfilter_size):
    # wav read for VAD
    vad_frame_index = []
    sample_rate, wav_data = scipy.io.wavfile.read(wav_path)
    vad = webrtcvad.Vad()
    vad.set_mode(vad_aggressive)
    frames = vadwav.frame_generator(vad_frame_size, wav_data, sample_rate, padding=True)
    frames = list(frames)
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        vad_frame_index.append(1 if is_speech else 0)

    if log_level > 0:
        print "LOG: apply median filter to result of vad"
    med_vad_frame_index = signal.medfilt(vad_frame_index,vad_medfilter_size)

    if log_level > 1:
        # check median filter
        # if the scale is too large, the graph is hard to see
        global fignum
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
        plt.plot(fr,med_vad_frame_index[0:endi])
        plt.axis([fr[0],fr[-1],-0.5,1.5])
        plt.title('After median filtering')


    if log_level > 0:
        print 'LOG: extend of vad result from frame unit to sample unit'
    # repeat matrix [window_sample * frame_size]
    med_vad_frame_index_rep = np.matlib.repmat(med_vad_frame_index,len(frame.bytes),1)
    [row,col] = med_vad_frame_index_rep.shape
    # 1 frame -> N window_sample
    out_vad_index = np.reshape(np.transpose(med_vad_frame_index_rep),[row*col,])

    # remove padded data
    if row*col > len(wav_data):
        if log_level > 0:
            print 'LOG: remove padded data'
        pad_size = row*col - len(wav_data)
        out_vad_index = out_vad_index[0:len(out_vad_index)-pad_size]

    return out_vad_index, wav_data

def is_voice_frame(vad_index,proportion=0.1):
    fsize = len(vad_index)
    sdata = np.sum(vad_index)
    if sdata < fsize*proportion:
        return False
    else:
        return True

def output_data(filename, data, info_file=""):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

    if not info_file=="":
        fi = open(info_file, 'a')
        fi.write("%s [ %d %d ] [ %d ] %s\n"
                 %(filename,data[0].shape[0],data[0].shape[1],data[1].shape[0], data[2]))


def spec_zm(spec_data):
    ntime = spec_data.shape[1]
    mean_data = np.matlib.repmat(np.mean(spec_data,axis=1),ntime,1)
    zm_data = spec_data - np.transpose(mean_data)
    return zm_data


def main():

    usage = "%prog [options] <wav-file> <out-file>"
    parser = OptionParser(usage)

    # parser.add_option('--spec-type', dest='spec_type', help='spectrogram type  [default: scispec ]',
    #                   default='scispec', type='string')
    parser.add_option('--sample-frequency', dest='sample_rate', help='sample rate of wav  [default: 16kHz ]',
                      default=16000, type='int')
    parser.add_option('--frame-length', dest='frame_size', help='frame size (ms)  [default: 25ms ]',
                      default=25, type='int')
    parser.add_option('--frame-shift', dest='frame_shift', help='frame shift size (ms)  [default: 10ms ]',
                      default=10, type='int')
    parser.add_option('--fft-size', dest='fft_size', help='fft size [default: frame size ]',
                      default=-1, type='int')

    parser.add_option('--vad-aggressive', dest='vad_agg', help='aggressive number for VAD (0~3 / least ~ most) [default: 3 ]',
                      default=3, type='int')
    parser.add_option('--vad-frame-size', dest='vad_frame_size',
                      help='frame size (10, 20, 30 ms) for VAD [default: 10ms ]',
                      default=10, type='int')
    parser.add_option('--vad-medfilter-size', dest='vad_medfilter_size',
                      help='median filter size for VAD [default: 5 frame]',
                      default=5, type='int')

    parser.add_option('--target-label', dest='target_label', help='target label  [default: None ]',
                      default='None', type='string')
    parser.add_option('--save-length', dest='save_length', help='maximum length to save (sec) [default: wav-length ]',
                      default=-1, type='int')
    parser.add_option('--save-info', dest='save_info_file', help='save information file [ optional ]',
                      default="", type='string')

    (o, args) = parser.parse_args()
    (wav_path, out_file) = args

    sr_ = o.sample_rate
    frame_size_ = np.int(o.frame_size * sr_ * 0.001)
    frame_shift_ = np.int(o.frame_shift * sr_ * 0.001)
    # splice_size = o.splice_size # data size = splice_size * 2 + 1 / data : left_splice + center + right_splice
    vad_aggressive = o.vad_agg # 0 ~ 3 (least ~ most agrressive)
    vad_frame_size = o.vad_frame_size # only 10 or 20 or 30 (ms)
    vad_medfilter_size = o.vad_medfilter_size # N frame
    target_label = o.target_label # speech / music / noise
    info_file = o.save_info_file
    global fignum

    if o.fft_size == -1:
        fft_size_ = frame_size_
    else:
        fft_size_ = o.fft_size

    if o.save_length == -1:
        nlen_save = o.save_length  # maximum length to save (sec)
    else:
        nlen_save = int(o.save_length * 1000 / o.frame_shift)


    # segment_time is center of each frame / spec_data = [frequncy x time]
    if log_level > 0:
        print 'LOG: extract spectrogram data'
    sample_freq, segment_time, spec_data = extract_spec.log_spec_scipy(wav_path, sr_, frame_size_, frame_shift_, fft_size_)
    if log_level > 1:
        fig = plt.figure(fignum)
        fignum += 1
        fig.suptitle('Spectrogram')
        if spec_data.shape[1] > 500:
            endi = 500
        else:
            endi = spec_data.shape[1]

        plt.pcolormesh(segment_time[0:endi],sample_freq,spec_data[:,0:endi])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

    # normalize zero mean
    spec_data_zm = spec_zm(spec_data)
    if log_level > 1:
        fig = plt.figure(fignum)
        fignum += 1
        fig.suptitle('Spectrogram with zero mean')
        if spec_data.shape[1] > 500:
            endi = 500
        else:
            endi = spec_data.shape[1]

        plt.pcolormesh(segment_time[0:endi],sample_freq,spec_data_zm[:,0:endi])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

    # Padding copy to first and last frame ( if splice_size is 2, then [1,2,3] -> [1,1,1,2,3,3,3] )
    # if log_level > 0:
    #     print 'LOG: padding spectrogram data for splicing with edge-mode'
    # pad_data = np.pad(spec_data,((0,0),(splice_size,splice_size)),'edge')

    # Voice activity detector using 'webrtcvad'
    if log_level > 0:
        print 'LOG: precessing vad'
    vad_index, wav_data = decision_vad(wav_path, vad_aggressive, vad_frame_size, vad_medfilter_size)

    if log_level > 1:
        fig = plt.figure(fignum)
        fignum += 1
        # convert to range : -32768 ~ 32767 -> -1 ~ 1
        wav_data = np.array(wav_data) / float(2 ** 15 - 1)
        cheak_time = 10 # sec
        if len(wav_data) > cheak_time*sr_:
            endi = cheak_time*sr_
        else:
            endi = len(wav_data)
        t = np.array(range(0,endi))/float(sr_)
        fig.suptitle('Check VAD result')
        plt.plot(t,wav_data[0:endi])
        plt.hold(True)
        plt.plot(t,vad_index[0:endi],'r')
        plt.hold(False)
        plt.axis([t[0],t[-1],-1,1.2])
        plt.xlabel('sec')
        plt.legend(['wav','vad'], loc=4)

    # make vad_data
    if log_level > 0:
        print 'LOG: match between spec_data and vad_index'
    vad_data = np.zeros(spec_data.shape[1])
    for i in range(len(segment_time)):
        center_sample = int(segment_time[i] * sr_)
        begi = center_sample-int(frame_size_/2)
        endi = center_sample+int(frame_size_/2)
        vadi = vad_index[begi:endi]
        vad_data[i] = (1 if is_voice_frame(vadi) else 0)

    if nlen_save == -1:
        output_data(out_file,(spec_data_zm,vad_data,target_label),info_file)
    else:
        nsplit = int(spec_data_zm.shape[1] / nlen_save) + 1
        out_file_str = out_file.split('.')
        out_file_name = out_file_str[0]
        out_file_ext = out_file_str[1]

        for i in range(nsplit):
            begi = i*nlen_save
            endi = (i+1)*nlen_save
            if endi > spec_data_zm.shape[1]:
                endi = spec_data_zm.shape[1]
            out_file_split = "%s_%d.%s"%(out_file_name,i,out_file_ext)
            out_spec_data = spec_data_zm[:,begi:endi]
            out_vad_data = vad_data[begi:endi]
            output_data(out_file_split,(out_spec_data,out_vad_data,target_label),info_file)



    if log_level > 1:
        plt.show()

if __name__=="__main__":
    main()
