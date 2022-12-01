import os
import numpy as np
import argparse
import logging
from matplotlib.pyplot import flag 
import torch 
import torch.nn as nn
import pickle
from sklearn import preprocessing
import time
from utils import loading_cv_data, cca_metric_derivative, AttentionFusion, TransformLayers, DCCA_AM
from utils import batch_size, emotion_categories, epochs, eeg_input_dim, eye_input_dim, output_dim,  learning_rate
from utils import EarlyStopping

def create_log_dir():
    time_ = time.strftime("%Y%m%d_%H%M%S")
    path_log = './logs/' + time_
    path_trained_model = './trained_model/' + time_ + '/'
    flag1, flag2 = os.path.exists(path_log), os.path.exists(path_trained_model)
    if not flag1:
        os.makedirs(path_log)
    if not flag2:
        os.makedirs(path_trained_model)

def run_nn(eeg_directory, eye_directory, lr=None, bs=None, gpu):
    if lr != None:
        learning_rate = lr
   
    if bs != None:
        batch_size = bs
    
    if gpu != "cuda:1":
        gpu = "cuda:0"

    cv = 3

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    print('Using:', device)
    eeg_dir = eeg_directory
    eye_dir = eye_directory
    file_list = os.listdir(eeg_dir)
    file_list.sort()
    time_ = time.strftime("%Y%m%d_%H%M%S")

    path_log = './logs/' + time_
    path_trained_model = './trained_model/' + time_ + '/'

    train_loss = []
    validation_loss = []
    i = 0
    for f_id in file_list:
        
        type(f_id)
        logging.basicConfig(filename=path_log+ '/cv3.log', level=logging.DEBUG)
        logging.debug('{}'.format(f_id))
        logging.debug('Task-Epoch-CCALoss-PredicLoss-PredicAcc')
        if f_id.endswith('.npz'):
            print(f_id)
            train_all, test_all = loading_cv_data(eeg_dir, eye_dir, f_id, cv)
            np.random.shuffle(train_all)
            np.random.shuffle(test_all)

            sample_num = train_all.shape[0]
            batch_number = sample_num // batch_size

            train_eeg = train_all[:, 0:310]
            train_eye = train_all[:, 310:343]
            train_label = train_all[:, -1]

            scaler = preprocessing.MinMaxScaler()
            train_eeg = scaler.fit_transform(train_eeg)
            train_eye = scaler.fit_transform(train_eye)
            test_eeg = test_all[:, 0:310]
            test_eye = test_all[:, 310:343]
            test_label = test_all[:, -1]

            test_eeg = scaler.fit_transform(test_eeg)
            test_eye = scaler.fit_transform(test_eye)

            train_eeg = torch.from_numpy(train_eeg).to(torch.float).to(device)
            train_eye = torch.from_numpy(train_eye).to(torch.float).to(device)
            test_eeg = torch.from_numpy(test_eeg).to(torch.float).to(device)
            test_eye = torch.from_numpy(test_eye).to(torch.float).to(device)
            train_label = torch.from_numpy(train_label).to(torch.long).to(device)
            test_label = torch.from_numpy(test_label).to(torch.long).to(device)

            for hyper_choose in range(50):

                    best_test_res = {}
                    best_test_res['acc'] = 0
                    best_test_res['predict_proba'] = None
                    best_test_res['fused_feature'] = None
                    best_test_res['transformed_eeg'] = None
                    best_test_res['transformed_eye'] = None
                    best_test_res['alpha'] = None
                    best_test_res['true_label'] = None
                    best_test_res['layer_size'] = None
                    # try 100 combinations of different hidden units
                    layer_sizes = [np.random.randint(50,200), np.random.randint(50,200), output_dim]
                    logging.info('{}-{}'.format(layer_sizes[0], layer_sizes[1]))
                    print(layer_sizes)
                    mymodel = DCCA_AM(eeg_input_dim, eye_input_dim, layer_sizes, layer_sizes, output_dim, emotion_categories, device).to(device)
                    optimizer_classifier = torch.optim.RMSprop(mymodel.parameters(), lr=learning_rate)
                    optimizer_model1 = torch.optim.RMSprop(iter(list(mymodel.parameters())[0:8]), lr=learning_rate/2)
                    optimizer_model2 = torch.optim.RMSprop(iter(list(mymodel.parameters())[8:16]), lr=learning_rate/2)
                    class_loss_func = nn.CrossEntropyLoss()
                    for epoch in range(epochs):
                        mymodel.train()
                        best_acc = 0
                        total_classification_loss = 0
                        for b_id in range(batch_number+1):
                            if b_id == batch_number:
                                train_eeg_used = train_eeg[batch_size*batch_number:, :]
                                train_eye_used = train_eye[batch_size*batch_number: , :]
                                train_label_used = train_label[batch_size*batch_number:]
                            else:
                                train_eeg_used = train_eeg[b_id*batch_size:(b_id+1)*batch_size, :]
                                train_eye_used = train_eye[b_id*batch_size:(b_id+1)*batch_size, :]
                                train_label_used = train_label[b_id*batch_size:(b_id+1)*batch_size]

                            # predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, transformed_1, transformed_2, alpha  = mymodel(train_eeg_used, train_eye_used)
                            predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, alpha  = mymodel(train_eeg_used, train_eye_used)
                            predict_loss = class_loss_func(predict_out, train_label_used)

                            optimizer_model1.zero_grad()
                            optimizer_model2.zero_grad()
                            optimizer_classifier.zero_grad()

                            partial_h1 = torch.from_numpy(partial_h1).to(torch.float).to(device)
                            partial_h2 = torch.from_numpy(partial_h2).to(torch.float).to(device)

                            output1.backward(-0.1*partial_h1, retain_graph=True)
                            output2.backward(-0.1*partial_h2, retain_graph=True)
                            predict_loss.backward()

                            optimizer_model1.step()
                            optimizer_model2.step()
                            optimizer_classifier.step()

                        # for every epoch, evaluate the model on both train and test set
                        mymodel.eval()
                        predict_out_train, cca_loss_train, _, _, _, _, _, _  = mymodel(train_eeg, train_eye)
                        predict_loss_train = class_loss_func(predict_out_train, train_label)
                        accuracy_train = np.sum(np.argmax(predict_out_train.detach().cpu().numpy(), axis=1) == train_label.detach().cpu().numpy()) / predict_out_train.shape[0]

                        predict_out_test, cca_loss_test, output_1_test, output_2_test, _, _, fused_tensor_test, attention_weight_test  = mymodel(test_eeg, test_eye)
                        predict_loss_test = class_loss_func(predict_out_test, test_label)
                        accuracy_test = np.sum(np.argmax(predict_out_test.detach().cpu().numpy(), axis=1) == test_label.detach().cpu().numpy()) / predict_out_test.shape[0]
                        
                        train_loss.append(predict_loss_train)
                        validation_loss.append(predict_loss_test)

                        early_stopping = EarlyStopping(tolerance=5, min_delta=10)
                        early_stopping(predict_loss_train, predict_loss_test)
                        
                        i += 1 
                        if early_stopping.early_stop:
                            print("We are at epoch:", i)
                            break
                        
                        if accuracy_test > best_test_res['acc']:
                            print('Accuracy improved from {} to {}'.format(best_test_res['acc'], accuracy_test) )
                            best_test_res['acc'] = accuracy_test
                            best_test_res['layer_size'] = layer_sizes
                            best_test_res['predict_proba'] = predict_out_test.detach().cpu().data
                            best_test_res['fused_feature'] = fused_tensor_test
                            best_test_res['transformed_eeg'] = output_1_test.detach().cpu().data
                            best_test_res['transformed_eye'] = output_2_test.detach().cpu().data
                            best_test_res['alpha'] = attention_weight_test
                            best_test_res['true_label'] = test_label.detach().cpu().data
                            torch.cuda.empty_cache()
                            print('Cuda cache cleared!')

                        print('Btch size: {} -- Hyperchoose Itr: {} -- Epoch: {} -- Train CCA loss is: {} -- Train loss: {} -- Train accuracy: {}'.format(batch_size, hyper_choose, epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
                        print('Btch size: {} -- Hyperchoose Itr: {} -- Epoch: {} -- Test CCA loss is: {} -- Test loss: {} -- Test accuracy: {}'.format(batch_size, hyper_choose, epoch, cca_loss_test, predict_loss_test.data, accuracy_test))
                        print('\n')
                        logging.info('\tTrain\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
                        logging.info('\tTest\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_test, predict_loss_test.data, accuracy_test))

                    pickle.dump(best_test_res, open( os.path.join(path_trained_model, f_id[:-8]+'_'+str(hyper_choose)), 'wb'  ))
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training DCAA Model for Multi-Modal emotion recognition"
    )
    parser.add_argument(
        "--p1",
        required=True,
        help="Path to the directory with the EEG feature files")
    parser.add_argument(
        "--p2",
        required=True,
        help="Path to the directory with the Eye feature files")
    parser.add_argument(
        '--lr',
        default=False,
        required=False, 
        type=float,
        help="Enter a valid learning rate for the network")

    parser.add_argument(
        '--bs',
        default=False,
        required=False, 
        type=int,
        help="Enter a valid batch size for the network")

    parser.add_argument(
        '--gpu',
        default="cuda:1",
        required=False, 
        type=int,
        help="0 or 1")

    arg = parser.parse_args()
    eeg_directory = arg.p1
    eye_directory = arg.p2
    lr = arg.lr
    bs = arg.bs
    gpu = arg.gpu

    create_log_dir()
    run_nn(eeg_directory, eye_directory, lr, bs, gpu)

    # python3 DLER.py --p1 ~/deep-learning-emotion-recognition/Dataset/SEED-V/EEG_DE_features/ --p2 ~/deep-learning-emotion-recognition/Dataset/SEED-V/Eye_movement_features/ --lr 0.0005 --bs 300