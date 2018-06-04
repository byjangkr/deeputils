import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import extract_spec
from deeputils.vad import vadwav






def main():
    wavfile = '../sample_data/command_wav/on.wav'
    frame_size_ = 400
    sr_ = 16000
    sample_freq, segment_time, spec_data = extract_spec.log_spec_scipy(wavfile, 16000, 400, 160, 512)
    vad_index, wav_data = vadwav.decision_vad_index(wavfile, vad_aggressive=3, vad_frame_size=10, min_sil_frames=1)
    vad_data = np.zeros(spec_data.shape[1])
    for i in range(len(segment_time)):
        center_sample = int(segment_time[i] * sr_)
        begi = center_sample-int(frame_size_/2)
        endi = center_sample+int(frame_size_/2)
        vadi = vad_index[begi:endi]
        vad_data[i] = (1 if vadwav.is_voice_frame(vadi) else 0)


    mdl = TSNE(learning_rate=100)

    spec_data_zm = extract_spec.spec_zm(spec_data)
    feat_data = np.transpose(spec_data_zm)
    embd_data = mdl.fit_transform(feat_data)

    sphinx = (vad_data[:] == 1)
    silinx = (vad_data[:] == 0)

    em_sph = embd_data[sphinx,:]
    em_sil = embd_data[silinx,:]
    print em_sph.shape, em_sil.shape
    plt.scatter(em_sph[:,0],em_sph[:,1],color='r')
    plt.scatter(em_sil[:, 0], em_sil[:, 1], color='b')

    plt.show()



if __name__=="__main__":
    main()