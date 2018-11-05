import numpy as np
import time
import random
import h5py

def save_append_array(filename, array_):
    with open(filename,'ab') as f:
        pos = f.tell()
        np.save(f,array_)

    return pos

def save_append_h5py(filename, key_, array_, attr_key_ ,lab_):
    with h5py.File(filename, 'a') as hf:
        h = hf.create_dataset(key_, data=array_, dtype=np.float32)
        h.attrs[attr_key_] = lab_

def load_h5py_from_poslist(filename, poslist_):
    with h5py.File(filename,'r') as hf:
        datalen = int(len(poslist_))
        data_shape = np.array(hf.get(poslist_[0].strip())).shape
        shape_ = np.append(datalen, data_shape[:])

        _data_ary = np.zeros(shape_)
        _lab_list = ['None' for i in xrange(int(datalen))]

        for i in xrange(datalen):
            key_ = poslist_[i].strip()
            _data_ary[i] = hf.get(key_)
            _lab_list[i] = hf.get(key_).attrs["label"]

    return _data_ary, _lab_list


def load_array_from_pos(filename, pos_):
    with open(filename,'rb') as f:
        f.seek(pos_)
        _data = np.load(f)
        _pos = f.tell()

    return _data, _pos

def read_pos_lab_file(filename):
    pos_lab_list = []
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            if not line: break
            ipos = long(line.split(' ')[0])
            ilab = line.split(' ')[1].strip()
            pos_lab_list.append([ipos,ilab])

    return pos_lab_list

def read_dualpos_lab_file(filename):
    pos_lab_list = []
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            if not line: break
            ipos = long(line.split(' ')[0])
            ipos2 = long(line.split(' ')[1])
            ilab = line.split(' ')[2].strip()
            pos_lab_list.append([ipos,ipos2,ilab])

    return pos_lab_list

def read_pos_lab_vad_file(filename):
    pos_lab_list = []
    with open(filename,'r') as f:
        while True:
            line = f.readline()
            if not line: break
            ipos = long(line.split(' ')[0])
            ilab = line.split(' ')[1].strip()
            ivad = int(line.split(' ')[2])
            pos_lab_list.append([ipos,ilab,ivad])

    return pos_lab_list

def fast_load_array_from_poslist(filename, poslist_, array_shape_=0):
    with open(filename,'rb') as f:
        datalen = np.array(len(poslist_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_)
        poslist = np.array(poslist_[:],dtype=long)
        poslist.sort()

        i = 0
        pre_pos = f.tell()
        for pos_ in poslist:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)
            _data_ary[i] = np.load(f)
            pre_pos = f.tell()
            i += 1

    return _data_ary

def fast_load_array_from_pos_lab_list(filename, pos_lab_list_, array_shape_=0):
    # pos_lab_list : [ number_of_pos(string) label(string), ... ]
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_,dtype=np.float32)
        _lab_list = ['None' for i in xrange(int(datalen))]
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        i = 0
        pre_pos = f.tell()
        for pos_, lab_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)

            _data_ary[i] = np.load(f)
            _lab_list[i] = lab_
            pre_pos = f.tell()
            i += 1

    return _data_ary, _lab_list

def fast_load_array_from_object_with_auto_slice(filename, pos_lab_list_, splice_size_=50, stride_size_=50):
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        # shape_ = np.array([])
        # if array_shape_ == 0:
        #     tmp_ary = np.load(f)
        #     shape_ = np.append(datalen, tmp_ary.shape[:])
        # else:
        #     shape_ = np.append(datalen, array_shape_[:])
        #
        # _data_ary = np.zeros(shape_,dtype=np.float32)
        _lab_list = ['None' for i in xrange(int(datalen))]
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        _data_list = []
        _splice_index = []
        _lab_list = []

        i = 0
        pre_pos = f.tell()
        for pos_, lab_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)

            _data = np.load(f)
            _spec = _data.item().get('spec')
            _label = _data.item().get('label')
            spec_data_pad = np.pad(_spec, ((0, 0), (splice_size_, splice_size_)), 'edge')
            _data_list.append(spec_data_pad)

            begi = 0
            endi = begi + splice_size_ * 2 + 1
            while endi < spec_data_pad.shape[1]:
                # centeri = begi + splice_size_

                _splice_index.append([i, begi, endi])
                _lab_list.append(_label[begi])

                begi += stride_size_
                endi = begi + splice_size_ * 2 + 1


            pre_pos = f.tell()
            i += 1
            del _data, _spec, _label, spec_data_pad

        _splice_index_ary = np.array(_splice_index, dtype=int)

    return _data_list, _lab_list, _splice_index_ary


def fast_load_array_from_pos_lab_list_with_filter(filename, pos_lab_list_,filter,):
    # pos_lab_list : [ number_of_pos(string) label(string), ... ]
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        tmp_ary = np.load(f)
        shape_ = np.append(datalen, np.array([filter.shape[0], tmp_ary.shape[1]]))

        _data_ary = np.zeros(shape_,dtype=np.float32)
        _lab_list = ['None' for i in xrange(int(datalen))]
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        i = 0
        pre_pos = f.tell()
        for pos_, lab_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)

            _data_ary[i] = np.dot(filter,np.load(f))
            _lab_list[i] = lab_
            pre_pos = f.tell()
            i += 1

    return _data_ary, _lab_list

def fast_load_array_from_pos_lab_list_mulithread(filename, pos_lab_list_, array_shape_=0, nthread=1):
    # pos_lab_list : [ number_of_pos(string) label(string), ... ]
    # using multi-thread
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_)
        _lab_list = ['None' for i in xrange(int(datalen))]
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        i = 0
        pre_pos = f.tell()
        for pos_, lab_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)

            _data_ary[i] = np.load(f)
            _lab_list[i] = lab_
            pre_pos = f.tell()
            i += 1

    return _data_ary, _lab_list

def fast_load_array_from_dualpos_lab_list(filename, pos_lab_list_, array_shape_=0):
    # dualpos_lab_list : [ number_of_pos(string) number_of_pos2(string) label(string), ... ]
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
            tmp_ary = np.load(f)
            shape2_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_)
        _data_ary2 = np.zeros(shape2_)
        _lab_list = ['None' for i in xrange(int(datalen))]
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        i = 0
        pre_pos = f.tell()
        for pos_, pos2_ ,lab_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)
            _data_ary[i] = np.load(f)
            pre_pos = f.tell()

            mv_pos = pos2_ - pre_pos
            f.seek(mv_pos, 1)
            _data_ary2[i] = np.load(f)
            pre_pos = f.tell()

            _lab_list[i] = lab_
            i += 1

    return _data_ary, _data_ary2, _lab_list

def fast_load_array_from_pos_lab_vad_list(filename, pos_lab_list_, array_shape_=0):
    # pos_lab_list : [ number_of_pos(string) label(string), ... ]
    with open(filename,'rb') as f:
        datalen = np.array(len(pos_lab_list_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_)
        _lab_list = ['None' for i in xrange(int(datalen))]
        _vad_list = np.ones(datalen,dtype=int) * (-1)
        pos_lab_list = pos_lab_list_[:]
        pos_lab_list.sort()

        i = 0
        pre_pos = f.tell()
        for pos_, lab_, vad_ in pos_lab_list:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos,1)

            _data_ary[i] = np.load(f)
            _lab_list[i] = lab_
            _vad_list[i] = int(vad_)
            pre_pos = f.tell()
            i += 1

    return _data_ary, _lab_list, _vad_list

def data_splice_with_index(data_,inx_):

    ilist_inx, ibeg, iend = inx_[0]
    temp_data = data_[ilist_inx][:,ibeg:iend]
    _shape = np.append(np.array(len(inx_)), temp_data.shape)

    _data_ary = np.zeros(_shape,dtype=temp_data.dtype)

    i = 0
    for listinx, time_beg, time_end in inx_:
        _data_ary[i] = data_[listinx][:,time_beg:time_end]
        i += 1

    return _data_ary

def main():
    print 'Example code...'
    a = np.array([[1,2,3],[2,3,4]])
    b = a*2

    print a, b

    aryfile='../exp/arrayio.npy'
    pos_a = save_append_array(aryfile,a)
    pos_b = save_append_array(aryfile,b)

    print pos_a,pos_b

    poslist = [pos_a,pos_b]
    data_ary = fast_load_array_from_poslist(aryfile,poslist)
    # print data_ary

def ex_run_multithread():

    posfile = '/home2/byjang/project/music_detection/MD_fork/exp/cnn_spec_5class_nozmvn/egs/spec_data.pos'
    datfile = '/home2/byjang/project/music_detection/MD_fork/exp/cnn_spec_5class_nozmvn/egs/spec_data.npy'
    h5posfile = '/home2/byjang/project/music_detection/MD_fork/exp/cnn_spec_5class_nozmvn_h5py/egs/spec_data.pos'
    h5file = '/home2/byjang/project/music_detection/MD_fork/exp/cnn_spec_5class_nozmvn_h5py/egs/spec_data.npy'
    pos_lab_list = read_pos_lab_file(posfile)
    random.shuffle(pos_lab_list)

    with open(h5posfile,'r') as f:
        pos_lab_list_h5 = f.readlines()

    random.shuffle(pos_lab_list_h5)

    beg_time = time.strftime('%H%M%S')
    print "Start load - %s " % (beg_time)
    val_data, val_lab_list = fast_load_array_from_pos_lab_list(datfile, pos_lab_list[0:5000])
    end_time = time.strftime('%H%M%S')
    print "End load - %s " % (end_time)

    beg_time = time.strftime('%H%M%S')
    print "Start load - %s " % (beg_time)
    _, _ = load_h5py_from_poslist(h5file, pos_lab_list_h5[0:5000])

    end_time = time.strftime('%H%M%S')
    print "End load - %s " % (end_time)

    # hf = h5py.File(h5file,'w')
    # h5d1 = hf.create_dataset('data1',data=val_data[0],dtype=np.float32)
    # h5d1.attrs["label"] = val_lab_list[0]
    # hf.close()
    #
    # hf = h5py.File(h5file, 'a')
    # hf.create_dataset('data2', data=val_data[1])
    # hf.create_dataset('data2_lab', data=val_lab_list[1])
    # hf.close()
    #
    #
    #
    # hf = h5py.File(h5file, 'r')
    # d1 = np.array(hf.get('data1'))
    # d1_lab = hf.get('data1').attrs["label"]
    #
    # d2 = hf.get('data2')
    # d2_lab =  hf.get('data1').attrs["label"]


    # print d1[0], d1_lab
    # print d2[0], d2_lab







if __name__=="__main__":
    # main()
    ex_run_multithread()

