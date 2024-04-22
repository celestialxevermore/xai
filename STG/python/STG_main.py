from stg import STG
import argparse 
import sys 
import logging 
import numpy as np
import scipy.stats # for creating a simple dataset 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from dataset import create_twomoon_dataset
import torch

def parse_configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str,choices=['classification','regression'])
    parser.add_argument('--feature_selection',type=bool,default=True)
    parser.add_argument('--do_train',type=bool,default=True)
    parser.add_argument('--n_size',type=int,default=1000)
    parser.add_argument('--p_size',type=int,default=20)

    arguments = parser.parse_args()
    return arguments 

def init_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def init_model(args,X_train,device):
    '''
        if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None
    '''
    model = STG(task_type=args.task,input_dim=X_train.shape[1], output_dim=2, hidden_dims=[60, 20], activation='tanh',
    optimizer='SGD', learning_rate=0.1, batch_size=X_train.shape[0], feature_selection=args.feature_selection, sigma=0.5, lam=0.5, random_state=1, device=device) 
    return model 


if __name__=='__main__':
    logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
    args = parse_configuration()
    logging.info('Looking for configuration files')
    logging.info('Loading original dataset')
    X_data, y_data=create_twomoon_dataset(args.n_size,args.p_size)
    logging.info('Loading training dataset')
    logging.info('Loading validation dataset')
    if args.do_train:
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.3)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)
    logging.info('Building the model')

    device = str(init_device(args))
    # print('############### device :',str(device))
    model = init_model(args,X_train,device)

    model.fit(X_train, y_train, nr_epochs=6000, valid_X=X_valid, valid_y=y_valid, print_interval=1000)

    y_pred=model.predict(X_data)
    model.save_checkpoint('trained_model.pt')