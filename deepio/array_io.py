import numpy as np
import time
import random

def save_append_array(filename, array_):
    with open(filename,'ab') as f:
        pos = f.tell()
        np.save(f,array_)

    return pos

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
    pos_lab_list = read_pos_lab_file(posfile)
    random.shuffle(pos_lab_list)

    beg_time = time.strtime('%H%M%S')
    print "Start load - %s " % (beg_time)
    val_data, val_lab_list = fast_load_array_from_pos_lab_list(datfile, pos_lab_list[0:26619])

    end_time = time.strtime('%H%M%S')
    print "End load - %s " % (end_time)






if __name__=="__main__":
    main()

