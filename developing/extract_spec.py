"""
    Copyright 2018.4. Byeong-Yong Jang
    byjang@cbnu.ac.kr
    This code is for extracting spectrogram from wav-file.

    Usage in sh
    -----------
    python extract_spec.py --spec-type=scispec --sample-frequency=16000 \
                           --frame-length=25 --frame-shift=10 --fft-size=512 \
                           --plot-spec=True \
                           test.wav  test.bin

   Usage in python
   ---------------
   import extract_spec as exspec
   spec_data = exspec.run('test.wav','test.bin','scispec',16000,25,10,512)


    Input
    -----
    wav-file : path of wave file 
    out-file : path of output file (pickle format)


    Options
    -------
    --spec-type (string) : type of spectrogram
        tfspec      - compute log-spectrogram using 'tensorflow'
        tfmfcc      - compute mfcc using 'tensorflow'
        scispec     - compute log-spectrogram using 'scipy' (default)
        rosaspec    - compute log-spectrogram using 'librosa'
        rosamelspec - compute mel-scale spectrogram using 'librosa'
        rosachroma  - compute chroma spectrogram using 'librosa'

    --sample-frequency (int) : sample rate of wav (Hz)
    --frame-length (int) : frame size (ms)
    --frame-shift (int) : frame shift size (ms)
    --fft-size (int) : FFT window size (sample)
    --plot-spec (True or any) : plot spectrogram using matplotlib

"""

from optparse import OptionParser

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

### hyper parameters ###
log_offset = 1e-6

### end of hyper parameter ###

def check_sample_rate(wavfile,input_rate, wavfile_rate):
    if (input_rate != wavfile_rate) :
        print "WARNING(extract_spec.py) : sample rate of wav is %d -> %s\n" %(wavfile_rate, wavfile)

# output spec data to pickle file
def output_data(filename,data):
    f = open(filename,'wb')
    pickle.dump(data,f)
    f.close()

# compute log-spectrogram using 'tensorflow' : 'tfspec'
def log_spec_tensorflow(wavfile, _sr, frame_size, frame_shift):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string,[])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    wav_sample_rate = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wavfile}).sample_rate
    check_sample_rate(wavfile, _sr, wav_sample_rate)

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    log_spectrogram = tf.log(spectrogram[0] + log_offset)

    log_spec_data = sess.run(log_spectrogram, feed_dict={wav_filename_placeholder: wavfile})

    return np.transpose(log_spec_data)


# compute mfcc using 'tensorflow' : tfmfcc
def mfcc_tensorflow(wavfile, _sr, frame_size, frame_shift, order=13):
    sess = tf.InteractiveSession()
    wav_filename_placeholder = tf.placeholder(tf.string, [])

    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    wav_data = wav_decoder.audio

    wav_sample_rate = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wavfile}).sample_rate
    check_sample_rate(wavfile, _sr, wav_sample_rate)

    spectrogram = contrib_audio.audio_spectrogram(
        wav_data,
        window_size=frame_size,
        stride=frame_shift,
        magnitude_squared=True)

    mfcc_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate, dct_coefficient_count=order)

    mfcc_data= sess.run(mfcc_, feed_dict={wav_filename_placeholder: wavfile})

    return mfcc_data


# compute log-spectrogram using 'scipy' : 'scispec'
def log_spec_scipy(wavfile, _sr, frame_size, frame_shift, fft_size):
    sample_rate, data = scipy.io.wavfile.read(wavfile)
    check_sample_rate(wavfile,_sr,sample_rate)
    # if nfft is 'None', fft size is 'nperseg'
    sample_freq, segment_time, spec_data = scipy.signal.spectrogram(data, fs=sample_rate,
                                                                    window='hann', nperseg=frame_size,
                                                                    noverlap=(frame_size - frame_shift),
                                                                    nfft=fft_size,
                                                                    scaling='spectrum',
                                                                    detrend=False,
                                                                    mode='psd')
    # if mode='magnitude', then the option of stft is follow as :
    # _, t, S = scipy.signal.stft(data,fs=sample_rate,nperseg=frame_size_,noverlap=(frame_size_-frame_shift_),
    #                             nfft=fft_size_,padded=False,boundary=None)
    # mode = {psd, complex, magnitude, angle, phase}
    log_spec_data = np.log(spec_data + log_offset)
    return sample_freq, segment_time, log_spec_data


# compute log-spectrogram using 'librosa' : 'rosaspec'
def log_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    log_spec_data = np.log(np.abs(np.conj(spec_data)*spec_data*2) + log_offset)
    return np.transpose(log_spec_data)


# compute mel-scale spectrogram using 'librosa' : 'rosamelspec'
def mel_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(y=data,sr=fs,S=spec_data,
                                       n_mels=64,fmin=0.0,fmax=7600) # parameter for mel-filter
    log_S = np.log(S + log_offset)
    return np.transpose(log_S)

# compute chroma spectrogram using 'librosa' : 'rosachroma'
def chroma_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile, sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data, n_fft=fft_size, hop_length=frame_shift, win_length=frame_size,
                                  window='hann', center=False)
    chroma_data = librosa.feature.chroma_stft(sr=fs,S=spec_data,n_fft=512)
    return np.transpose(chroma_data)

def plot_spec(spec_data):
    plt.figure()
    plt.pcolormesh(spec_data)
    plt.title('Log spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

def run(wavfile_,outfile_,spec_type_='scispec',sample_rate_=16000,frame_size_=25,frame_shift_=10,fft_size_=512):

    wav_path = wavfile_
    out_file = outfile_

    spec_type = spec_type_
    sr_ = sample_rate_
    frame_size_ = np.int(frame_size_ * sr_ * 0.001)
    frame_shift_ = np.int(frame_shift_ * sr_ * 0.001)
    fft_size = fft_size_

    if spec_type == 'tfspec':
        spec_data = log_spec_tensorflow(wav_path,sr_,frame_size_,frame_shift_)
    elif spec_type == 'tfmfcc':
        spec_data = mfcc_tensorflow(wav_path,sr_,frame_size_,frame_shift_,order=13)
    elif spec_type == 'scispec':
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosaspec':
        spec_data = log_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosamelspec':
        spec_data = mel_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosachroma':
        spec_data = chroma_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    else:
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size_)

    return spec_data


def main():

    usage = "%prog [options] <wav-file> <out-file>"
    parser = OptionParser(usage)

    parser.add_option('--spec-type', dest='spec_type', help='spectrogram type  [default: scispec ]',
                      default='scispec', type='string')
    parser.add_option('--sample-frequency', dest='sample_rate', help='sample rate of wav  [default: 16kHz ]',
                      default=16000, type='int')
    parser.add_option('--frame-length', dest='frame_size', help='frame size (ms)  [default: 25ms ]',
                      default=25, type='int')
    parser.add_option('--frame-shift', dest='frame_shift', help='frame shift size (ms)  [default: 10ms ]',
                      default=10, type='int')
    parser.add_option('--fft-size', dest='fft_size', help='fft size [default: frame size ]',
                      default=-1, type='int')
    parser.add_option('--plot-spec', dest='plot_spec', help='plot spectrogram  [default: False ]',
                      default='False', type='string')

    (o, args) = parser.parse_args()
    (wav_path, out_file) = args

    spec_type = o.spec_type
    sr_ = o.sample_rate
    frame_size_ = np.int(o.frame_size * sr_ * 0.001)
    frame_shift_ = np.int(o.frame_shift * sr_ * 0.001)

    if o.fft_size == -1:
        fft_size_ = frame_size_
    else:
        fft_size_ = o.fft_size

    if spec_type == 'tfspec':
        spec_data = log_spec_tensorflow(wav_path,sr_,frame_size_,frame_shift_)
    elif spec_type == 'tfmfcc':
        spec_data = mfcc_tensorflow(wav_path,sr_,frame_size_,frame_shift_,order=13)
    elif spec_type == 'scispec':
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosaspec':
        spec_data = log_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosamelspec':
        spec_data = mel_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosachroma':
        spec_data = chroma_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    else:
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size_)

    if o.plot_spec == 'True':
        plt.figure()
        plt.pcolormesh(spec_data)
        plt.title('Log spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.show()

    output_data(out_file,spec_data)

if __name__=="__main__":
    main()
