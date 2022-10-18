import os
import pickle
import numpy as np

def loading_cv_data(eeg_dir, eye_dir, file_name, cv_number):
    eeg_data_pickle = np.load( os.path.join(eeg_dir, file_name))
    eye_data_pickle = np.load( os.path.join(eye_dir, file_name))
    eeg_data = pickle.loads(eeg_data_pickle['data'])
    eye_data = pickle.loads(eye_data_pickle['data'])
    label = pickle.loads(eeg_data_pickle['label'])
    list_1 = [0,1,2,3,4,15,16,17,18,19,30,31,32,33,34]
    list_2 = [5,6,7,8,9,20,21,22,23,24,35,36,37,38,39]
    list_3 = [10,11,12,13,14,25,26,27,28,29,40,41,42,43,44]
    if cv_number == 1:
        print('#1 as test, preparing data')
        train_list = list_2 + list_3
        test_list = list_1
    elif cv_number == 2:
        print('#2 as test, preparing data')
        train_list = list_1 + list_3
        test_list = list_2
    else:
        print('#3 as test, preparing data')
        train_list = list_1 + list_2
        test_list = list_3

    train_eeg = []
    test_eeg = []
    train_label = []
    for train_id in range(len(train_list)):
        train_eeg_tmp = eeg_data[train_list[train_id]]
        train_eye_tmp = eye_data[train_list[train_id]]
        train_label_tmp = label[train_list[train_id]]
        if train_id == 0:
            train_eeg = train_eeg_tmp
            train_eye = train_eye_tmp
            train_label = train_label_tmp
        else:
            train_eeg = np.vstack((train_eeg, train_eeg_tmp))
            train_eye = np.vstack((train_eye, train_eye_tmp))
            train_label = np.hstack((train_label, train_label_tmp))
    assert train_eeg.shape[0] == train_eye.shape[0]
    assert train_eeg.shape[0] == train_label.shape[0]

    test_eeg = []
    test_eye = []
    test_label = []
    for test_id in range(len(test_list)):
        test_eeg_tmp = eeg_data[test_list[test_id]]
        test_eye_tmp = eye_data[test_list[test_id]]
        test_label_tmp = label[test_list[test_id]]
        if test_id == 0:
            test_eeg = test_eeg_tmp
            test_eye = test_eye_tmp
            test_label = test_label_tmp
        else:
            test_eeg = np.vstack((test_eeg, test_eeg_tmp))
            test_eye = np.vstack((test_eye, test_eye_tmp))
            test_label = np.hstack((test_label, test_label_tmp))
    assert test_eeg.shape[0] == test_eye.shape[0]
    assert test_eeg.shape[0] == test_label.shape[0]

    train_all = np.hstack((train_eeg, train_eye, train_label.reshape([-1,1])))
    test_all = np.hstack((test_eeg, test_eye, test_label.reshape([-1,1])))
    return train_all, test_all