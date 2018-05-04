#!/usr/bin/env python


import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

### hyper parameters ###
bool_plot = True
log_offset = 1e-6

### end of hyper parameter ###

# compute log-spectrogram using 'tensorflow'
def log_spec_tensorflow(wavfile,frame_size=400,frame_shift=160):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string,[])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    log_spectrogram = tf.log(spectrogram[0] + log_offset)

    log_spec_data = sess.run(log_spectrogram, feed_dict={wav_filename_placeholder: wavfile})

    return log_spec_data


# compute mfcc using 'tensorflow'
def mfcc_tensorflow(wavfile, frame_size=400, frame_shift=160 , order=13):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string, [])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    mfcc_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=order)

    mfcc_data= sess.run(mfcc_, feed_dict={wav_filename_placeholder: wavfile})

    return mfcc_data


# compute log-spectrogram using 'scipy'
def log_spec_scipy(wavfile,frame_size=400,frame_shift=160):
    sample_rate, data = scipy.io.wavfile.read(wavfile)
    # if nfft is 'None', fft size is 'nperseg'
    sample_freq, segment_time, spec_data = scipy.signal.spectrogram(data, fs=sample_rate,
                                                                    window='hann', nperseg=frame_size,
                                                                    noverlap=(frame_size - frame_shift), nfft=512,
                                                                    mode='psd')
    # mode = {psd, complex, magnitude, angle, phase}
    log_spec_data = np.log(spec_data + log_offset)
    return sample_freq, segment_time, log_spec_data


# compute log-spectrogram using 'librosa'
def log_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile,sr=None)
    spec_data = librosa.core.stft(data,n_fft=512,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    log_spec_data = np.log(np.abs(np.conj(spec_data)*spec_data*2) + log_offset)
    return np.array(log_spec_data)


def mel_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile,sr=None)
    spec_data = librosa.core.stft(data,n_fft=512,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(y=data,sr=fs,S=spec_data,
                                       n_mels=64,fmin=0.0,fmax=7600) # parameter for mel-filter
    log_S = np.log(S + log_offset)
    return log_S

def chroma_spec_librosa(wavfile,frame_size=400,frame_shift=160):
    data, fs = librosa.load(wavfile, sr=None)
    spec_data = librosa.core.stft(data, n_fft=512, hop_length=frame_shift, win_length=frame_size,
                                  window='hann', center=False)
    chroma_data = librosa.feature.chroma_stft(sr=fs,S=spec_data,n_fft=512)
    return chroma_data

def main():

    # processing to script
    data_dir = '../sample_data/command_wav/'
    search_path = os.path.join(data_dir,'*.wav')
    file_list = gfile.Glob(search_path)

    data_ary = [] # data restore
    for wav_path in file_list[1:2]:

        log_spec_tf_data = log_spec_tensorflow(wav_path)
        mfcc_tf_data = mfcc_tensorflow(wav_path,order=13)
        sample_freq, segment_time, log_spec_scipy_data = log_spec_scipy(wav_path)
        # _, _, log_spec_scipy_data = log_spec_scipy(wav_path)

        log_spec_librosa_data = log_spec_librosa(wav_path)
        mel_spec_librosa_data = mel_spec_librosa(wav_path)
        chroma_spec_librosa_data = chroma_spec_librosa(wav_path)

        # data_ary.append(wanted_data)

    # cheak results
    print np.transpose(log_spec_tf_data).shape, log_spec_scipy_data.shape, log_spec_librosa_data.shape
    print mel_spec_librosa_data.shape, chroma_spec_librosa_data.shape
    print np.transpose(log_spec_tf_data)[10][0:5]
    print log_spec_scipy_data[10][0:5]
    print log_spec_librosa_data[10][0:5]


    if bool_plot:
        plt.figure(1)
        plt.pcolormesh(segment_time, sample_freq, np.transpose(log_spec_tf_data))
        plt.title('Log spectrogram (direct) of tensorflow')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.figure(2)
        plt.pcolormesh(segment_time, sample_freq, log_spec_scipy_data)
        plt.title('Log spectrogram of scipy')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.figure(3)
        rosa_shape = np.transpose(log_spec_librosa_data).shape
        plt.pcolormesh(segment_time[0:rosa_shape[0]],sample_freq,log_spec_librosa_data)
        plt.title('Log spectrogram of librosa')

        plt.figure(4)
        mel_rosa_shape = np.transpose(mel_spec_librosa_data).shape
        plt.pcolormesh(segment_time[0:mel_rosa_shape[0]],np.array(range(0,mel_rosa_shape[1])),mel_spec_librosa_data)
        plt.colorbar()
        plt.title('Mel-scale spectrogram of librosa')

        plt.figure(5)
        chroma_rosa_shape = np.transpose(chroma_spec_librosa_data).shape
        plt.pcolormesh(segment_time[0:mel_rosa_shape[0]],np.array(range(0,chroma_rosa_shape[1])),chroma_spec_librosa_data)
        plt.colorbar()
        plt.title('Chroma spectrogram of librosa')

        plt.show()

if __name__=="__main__":
    main()
