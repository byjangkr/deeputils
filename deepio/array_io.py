import numpy as np

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



if __name__=="__main__":
    main()

