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
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import tensorflow as tf
from numpy import matlib
from scipy import signal
from scipy.io import wavfile
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops

### hyper parameters ###
log_offset = 1e-6

### end of hyper parameter ###
def freq_to_mel(freq_):
    return 2595.0*np.log10(1.0 + freq_/700.0)

def mel_to_freq(mel_):
    return (10**(mel_/2595.0) - 1.0)*700.0

def mel_scale_range(fft_size_,sr_,n_mel_=64):
    melfilt = librosa.filters.mel(sr=sr_,n_fft=fft_size_,n_mels=n_mel_)
    melscale_inx = np.zeros((n_mel_,melfilt.shape[1]),dtype=bool)
    melscale_dim = np.zeros(n_mel_,dtype=int)

    for i in xrange(n_mel_):
        melscale_inx[i] = (melfilt[i] > 0)
        melscale_dim[i] = np.count_nonzero(melfilt[i])


    # minfreq = 0
    # maxfreq = sr_ / 2.0
    #
    # fft_size = fft_size_
    # fftstep = (sr_ / 2.0) / fft_size
    # fft2freq = np.arange(minfreq,maxfreq,fftstep)
    #
    # melstep = (freq_to_mel(maxfreq) - freq_to_mel(minfreq)) / n_mel_
    # mel_range = np.arange(freq_to_mel(minfreq),freq_to_mel(maxfreq),melstep)
    # melfreq_range = mel_to_freq(mel_range)
    #
    # melscale_inx = np.zeros((n_mel_,fft_size),dtype=bool)
    # melscale_dim = np.zeros(n_mel_,dtype=int)
    # fftrange = np.arange(fft_size)
    #
    # for i in xrange(n_mel_):
    #     if i == (n_mel_-1): # last bin
    #         inx = (melfreq_range[i] <= fft2freq)
    #     else:
    #         inx = ((melfreq_range[i] <= fft2freq) & (melfreq_range[i+1] > fft2freq))
    #
    #     melscale_inx[i] = inx
    #     if not fftrange[inx].any():
    #         assert('ERROR(mel_scale_range): %d mel-bin is empty, reduce number of mel' %(i))
    #
    #     melscale_dim[i] = int(fftrange[inx].shape[0])
    return melscale_inx, melscale_dim, melfilt

def spec_zm(spec_data):
    ntime = spec_data.shape[1]
    mean_data = np.matlib.repmat(np.mean(spec_data,axis=1),ntime,1)
    zm_data = spec_data - np.transpose(mean_data)
    return zm_data

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
    if len(data.shape) > 1:
        data = (data[:,0] + data[:,1])/2
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
    # _, t, S = scipy.signal.stft(data[0:400],fs=sample_rate,nperseg=frame_size,noverlap=(frame_size-frame_shift),
    #                             nfft=400,padded=False,boundary=None)
    # mode = {psd, complex, magnitude, angle, phase}

    log_spec_data = np.log(spec_data + log_offset)

    return sample_freq, segment_time, log_spec_data


def segment_time_librosa(wav_length,fs,frame_size,frame_shift):
    beg_time = librosa.core.samples_to_time(np.arange(0,wav_length,frame_shift),sr=fs)
    segment_time = beg_time + (float(frame_size)/float(fs))/2.0
    return segment_time

# compute log-spectrogram using 'librosa' : 'rosaspec'
def log_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    log_spec_data = np.log(np.abs(np.conj(spec_data)*spec_data*2) + log_offset)

    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(log_spec_data.shape[1])]

    # spec_data1 = librosa.core.stft(data,n_fft=400,hop_length=160,win_length=400,
    #                               window='hann',center=False)
    # spec_data2 = librosa.core.stft(data[0:400], n_fft=400, hop_length=160, win_length=400,
    #                                window='hann', center=False)
    # print spec_data1[0:5,0]
    # print spec_data2[0:5]
    return  segment_time, log_spec_data


# compute mel-scale spectrogram using 'librosa' : 'rosamelspec'
def mel_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size, n_mels_=64, fmin_=0.0, fmax_=8000):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(sr=fs,S=(np.abs(spec_data)**2),
                                       n_mels=n_mels_,fmin=fmin_,fmax=fmax_) # parameter for mel-filter
    log_S = librosa.power_to_db(S)

    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(log_S.shape[1])]

    return segment_time, log_S

# compute mfcc using 'librosa' : 'rosamfcc'
def mfcc_librosa(wavfile, _sr, frame_size, frame_shift, fft_size, n_mels_=64, fmin_=0.0, fmax_=8000, n_mfcc_=13):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(sr=fs,S=(np.abs(spec_data)**2),
                                       n_mels=n_mels_,fmin=fmin_,fmax=fmax_) # parameter for mel-filter
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=n_mfcc_, dct_type=2)
    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(mfcc.shape[1])]

    return segment_time, mfcc

# compute mfcc using 'librosa' : 'rosamfccdel'
def mfccdel_librosa(wavfile, _sr, frame_size, frame_shift, fft_size, n_mels_=64, fmin_=0.0, fmax_=8000, n_mfcc_=13):
    data, fs = librosa.load(wavfile,sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data,n_fft=fft_size,hop_length=frame_shift,win_length=frame_size,
                                  window='hann',center=False)
    S = librosa.feature.melspectrogram(sr=fs,S=(np.abs(spec_data)**2),
                                       n_mels=n_mels_,fmin=fmin_,fmax=fmax_) # parameter for mel-filter
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S),n_mfcc=n_mfcc_, dct_type=2)
    mfcc_del1 = librosa.feature.delta(mfcc)
    mfcc_del2 = librosa.feature.delta(mfcc,order=2)

    mfccdel = np.zeros((mfcc.shape[0]*3,mfcc.shape[1]),dtype=mfcc.dtype)
    mfccdel[0:n_mfcc_,:] = mfcc[:]
    mfccdel[n_mfcc_*1:n_mfcc_*2,:] = mfcc_del1[:]
    mfccdel[n_mfcc_*2:n_mfcc_*3,:] = mfcc_del2[:]

    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(mfcc.shape[1])]

    return segment_time, mfccdel

# compute chroma spectrogram using 'librosa' : 'rosachroma'
def chroma_spec_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile, sr=None)
    check_sample_rate(wavfile,_sr,fs)
    spec_data = librosa.core.stft(data, n_fft=fft_size, hop_length=frame_shift, win_length=frame_size,
                                  window='hann', center=False)
    chroma_data = librosa.feature.chroma_stft(sr=fs,S=spec_data)

    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(chroma_data.shape[1])]

    return segment_time, chroma_data

# compute chroma spectrogram using 'librosa' : 'rosatempo'
def tempogram_librosa(wavfile, _sr, frame_size, frame_shift, fft_size):
    data, fs = librosa.load(wavfile, sr=None)
    check_sample_rate(wavfile,_sr,fs)
    tempo_data = librosa.feature.tempogram(data,sr=fs, hop_length=frame_shift, win_length=frame_size,
                                  window='hann', center=False)

    segtime = segment_time_librosa(len(data),fs,frame_size,frame_shift)
    segment_time = segtime[0:int(tempo_data.shape[1])]
    print tempo_data.shape, segtime.shape

    return segment_time, tempo_data

def plot_spec(spec_data):
    plt.figure()
    plt.pcolormesh(spec_data)
    plt.title('Log spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()

def run(wavfile_,spec_type_='scispec',sample_rate_=16000,frame_size_=25,frame_shift_=10,fft_size_=512):

    wav_path = wavfile_

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
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size)
    elif spec_type == 'rosaspec':
        _, spec_data = log_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size)
    elif spec_type == 'rosamelspec':
        _, spec_data = mel_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size)
    elif spec_type == 'rosachroma':
        _, spec_data = chroma_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size)
    else:
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size)

    return spec_data

def ex_run(wavfile_,spec_type_='scispec',sample_rate_=16000,frame_size_=25,frame_shift_=10,fft_size_=512):

    wav_path = wavfile_

    sr_ = sample_rate_
    frame_size_ = np.int(frame_size_ * sr_ * 0.001)
    frame_shift_ = np.int(frame_shift_ * sr_ * 0.001)
    fft_size = fft_size_

    _, segment_time, spec_data = log_spec_scipy(wav_path, sr_, frame_size_, frame_shift_, fft_size)
    # segment_time2, spec_data2 = tempogram_librosa(wav_path, sr_, frame_size_, frame_shift_, fft_size)
    melinx, meldim, _ = mel_scale_range(2048,sr_=16000,n_mel_=64)
    melfilt = librosa.filters.mel(sr=16000,n_fft=2048,n_mels=64)

    chromafilt12 = librosa.filters.chroma(sr=16000,n_fft=2048,n_chroma=12)
    chromafilt24 = librosa.filters.chroma(sr=16000, n_fft=2048, n_chroma=24)
    print chromafilt12[0], chromafilt24[0]

    ## figure melscale filter
    # fig = plt.figure(1)
    # for i in xrange(64):
    #     plt.plot(melfilt[i])

    ## figure chroma filter
    # fig = plt.figure(1)
    # for i in xrange(24):
    #     plt.plot(chromafilt24[i])

    ## figure chroma 12 VS 24
    # fig = plt.figure(1)
    # plt.plot(chromafilt12[0])
    # plt.plot(chromafilt12[1])
    # plt.plot(chromafilt24[0])
    # plt.plot(chromafilt24[1])
    # plt.legend(('12-0','12-1','24-0','24-1'))

    # plt.show()


    # print spec_data.shape, spec_data2.shape
    # print segment_time.shape, segment_time2.shape

    print "End of ex_run..."


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
        _, spec_data = log_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosamelspec':
        spec_data = mel_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosachroma':
        spec_data = chroma_spec_librosa(wav_path,sr_,frame_size_,frame_shift_,fft_size_)
    elif spec_type == 'rosamfcc':
        spec_data = mfcc_librosa(wav_path, sr_, frame_size_, frame_shift_, fft_size_)
    else:
        _, _, spec_data = log_spec_scipy(wav_path,sr_,frame_size_,frame_shift_,fft_size_)

    if o.plot_spec == 'True':
        plt.figure()
        plt.pcolormesh(spec_data)
        plt.title('Log spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        plt.show()

    #output_data(out_file,spec_data)

if __name__=="__main__":
    # main()
    ex_run('../sample_data/drama_sample_stereo.wav')
