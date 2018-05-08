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

def fast_load_array_from_poslist(filename, poslist_, array_shape_=0):
    with open(filename,'rb') as f:
        datalen = np.array(len(poslist_))

        if array_shape_ == 0:
            tmp_ary = np.load(f)
            shape_ = np.append(datalen, tmp_ary.shape[:])
        else:
            shape_ = np.append(datalen, array_shape_[:])

        _data_ary = np.zeros(shape_)
        poslist = poslist_[:]
        poslist.sort()

        i = 0
        pre_pos = 0
        for pos_ in poslist:
            mv_pos = pos_ - pre_pos
            f.seek(mv_pos)
            _data_ary[i] = np.load(f)
            pre_pos = pos_
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
    print data_ary



if __name__=="__main__":
    main()

