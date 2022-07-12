if __name__ == '__main__':
    import h5py
    import numpy as np

    train = h5py.File('./data/test_0.h5', 'r')
    # val = h5py.File('./data/h5/l/val_0.h5', 'r')
    # val_new = np.zeros((val['data'].shape[0], 14000, 3))
    for set in [train]:
        x_new = np.zeros((set['data'].shape[0], 10000, 3))
        y_new = np.zeros((set['data'].shape[0], 10000))
        x_orig = np.array(set['data'][:])
        y2_orig = np.array(set['label_seg'][:])  # your train set labels
        for i in range(x_orig.shape[0]):
            x_nonzero = x_orig[i, [not np.all(x_orig[i, j] == 0) for j in range(x_orig.shape[1])], :]
            y_nonzero = y2_orig[i, [not np.all(x_orig[i, j] == 0) for j in range(x_orig.shape[1])]]
            idx_num_list = {}
            idx_sort_list = list(np.unique(y_nonzero))
            idx_sort_list.sort(reverse=True)
            for idx in np.unique(y_nonzero):
                idx_num_list[idx] = len(np.where(y_nonzero == idx)[0]) / y_nonzero.shape[0]
            start = 0
            for idx in idx_sort_list:
                y_idx = y_nonzero[np.where(y_nonzero == idx)]
                x_idx = x_nonzero[np.where(y_nonzero == idx), :][0]
                if idx != 0:
                    idx_num = int(np.round(idx_num_list[idx] * 10000))
                    n = np.random.choice(len(y_idx), idx_num, replace=False)
                    x_idx_choiced = x_idx[n, :]
                    y_idx_choiced = y_idx[n]
                    x_new[i, start:start + idx_num] = x_idx_choiced
                    y_new[i, start:start + idx_num] = y_idx_choiced
                    start += idx_num
                else:
                    idx_num = 10000 - start
                    n = np.random.choice(len(y_idx), idx_num, replace=False)
                    x_idx_choiced = x_idx[n, :]
                    y_idx_choiced = y_idx[n]
                    x_new[i, start:start + idx_num] = x_idx_choiced
                    y_new[i, start:start + idx_num] = y_idx_choiced
                    start += idx_num
            print(i, x_new.shape, y_new.shape)

        if set == train:
            # an = {}
            # an['data'] = x_new
            # an['label_seg'] = y_new
            # np.save('./data/train_new.h5', an)
            file = h5py.File('./data/test_10000.h5', 'w')
            file.create_dataset('data', data=x_new)
            file.create_dataset('label_seg', data=y_new)
            file.close()
        elif set == val:
            # an = {}
            # an['data'] = x_new
            # an['label_seg'] = y_new
            # np.save('./data/val_new.h5', an)
            file = h5py.File('./data/val_10000.h5', 'w')
            file.create_dataset('data', data=x_new)
            file.create_dataset('label_seg', data=y_new)
            file.close()
