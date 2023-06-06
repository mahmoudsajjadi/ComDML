# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:18:29 2023

@author: seyedmahmouds
"""
import torch
import time
import argparse
import copy
from torch import nn
import sys

from model.resnet110_7t import resnet56_SFL_local_tier_7
from model.resnet110_7t import resnet56_SFL_fedavg_base

import os


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10




def add_args(parser):
    # Data loading and preprocessing related arguments
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--partition_alpha', type=float, default=1000000, metavar='PA',
                        help='partition alpha (default: 0.5)')
    parser.add_argument('--client_number', type=int, default=1, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)
    
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = add_args(parser)


# initialization

idx, lr, client_epoch = 0, 0.001, 1
criterion = nn.CrossEntropyLoss()

# dataset loader

def load_data(args, dataset_name): # only for cifar-10


    data_loader = load_partition_data_cifar10

    
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_number, args.batch_size)
    
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    return dataset


dataset = load_data(args, args.dataset)
[train_data_num, test_data_num, train_data_global, test_data_global,
 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

dataset_test = test_data_local_dict
dataset_train = train_data_local_dict


net_glob_client_tier, net_glob_server_tier = resnet56_SFL_local_tier_7(classes=class_num,tier=1)

def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch):
    global net_model_server, criterion, optimizer_server, device, batch_acc_train, batch_loss_train, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect, w_locals_server, w_glob_server, net_server, time_train_server_train, time_train_server_train_all, w_glob_server_tier, w_locals_server_tier, w_locals_tier
    global loss_train_collect_user, acc_train_collect_user, lr, total_time, times_in_server, new_lr
    global net_glob_server_tier

    time_train_server_s = time.time()
    


    net_server = copy.deepcopy(net_glob_server_tier).to(device)

        
    net_server.train()
    # lr = new_lr
    optimizer_server =  torch.optim.Adam(net_server.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code
    
    # train and update
    optimizer_server.zero_grad()
    
    fx_client = fx_client.to(device)
    y = y.to(device)
    
    #---------forward prop-------------
    fx_server = net_server(fx_client)
    
    # calculate loss
    # loss = criterion(fx_server, y)
    if args.dataset != 'HAM10000':
        y = y.to(torch.long)
        # y.int()
    loss = criterion(fx_server, y) # to solve change dataset
    
                    
    
    #--------backward prop--------------
    loss.backward()  #original
    dfx_client = fx_client.grad.clone().detach()
    # dfx_client = fx_client.grad.clone().detach()
    dfx_server = fx_server.clone().detach()
    optimizer_server.step()
    
    time_train_server = time.time() - time_train_server_s
    
    return time_train_server  

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = client_epoch
        #self.selected_clients = []
        batch_size = args.batch_size
        self.ldr_train = dataset_train[idx]
        self.ldr_test = dataset_test[idx]
            
        

    def train(self, tier, net):
        net.train()
        self.lr = lr
        


        optimizer_client =  torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.wd, amsgrad=True) # from fedgkt code

        # optimizer_client = optimizer_client_tier[idx]
        
        time_client=0
        data_transmited_sl_client = 0
        batch_size = args.batch_size
        CEloss_client_train = []

        for iter in range(self.local_ep):
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                time_s = time.time()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                
                
                    
                
                #---------forward prop-------------
                if tier != 0:
                    extracted_features, fx = net(images)
                    
                    client_fx = fx.clone().detach().requires_grad_(True)
                    
                    
                        
                    time_client += time.time() - time_s
                    
                    time_train_server = train_server(client_fx, labels, iter, self.local_ep, self.idx, len_batch)
                else:
                    extracted_features = net(images)
                    time_train_server = 0
                
                
                #--------backward prop -------------
                time_s = time.time()
                    
                labels = labels.to(torch.long)
                loss = criterion(extracted_features, labels) # to solve change dataset)
            
                loss.backward()

                optimizer_client.zero_grad()
                    
                optimizer_client.step()
                time_client += time.time() - time_s
                
                if tier != 0:
                    data_transmited_sl_client += (sys.getsizeof(client_fx.storage()) + 
                                          sys.getsizeof(labels.storage()))

       
        
        return net.state_dict(), time_client, data_transmited_sl_client, time_train_server 
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))  
    
def trainer(tier, net_glob_client_tier, net_glob_server_tier, w_glob_client_tier, dataset):
    

    
    net_glob_client_tier.to(device)
    
    
    
    
    # calculate the delay of server send model to clients
    data_server_to_client = 0
    data_transmited_fl = 0
    # for k in w_glob_client_tier[tier]:
    #     data_server_to_client += sys.getsizeof(w_glob_client_tier[tier][k].storage())
    # simulated_delay[idx] = data_server_to_client / net_speed[idx]
        # wandb.log({"Client{}_Tier".format(i): client_tier[i], "epoch": iter}, commit=False)
        
    data_transmited_fl_client = 0
    time_train_test_s = time.time()

    
    
    local = Client(net_glob_client_tier, idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = [], idxs_test = [])
        
    # Training ------------------
    
    [w_client, duration, data_transmited_sl_client, time_train_server] = local.train(tier, net = copy.deepcopy(net_glob_client_tier).to(device))
        
    
        
    first_side_time = duration
    second_side_time = time_train_server
    
    
    
    for k in w_client:
        data_transmited_fl_client = data_transmited_fl_client + sys.getsizeof(w_client[k].storage())
    data_transmited_fl += data_transmited_fl_client         
    
    data_transmitted_client = data_transmited_sl_client + data_transmited_fl_client
    
    # replace last observation with new observation for data transmission

    

    
    return first_side_time, second_side_time, data_transmitted_client
    
        

def tier_profiler(batch_size, dataset, num_tiers = 7) -> list[list]:
    '''
    

    Parameters
    ----------
    num_tiers : int
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.

    Returns
    -------
    [first_side_time, second_side_time, batch_data_size]
        DESCRIPTION.

    '''
    tier_sides_times_size = []
    first_side_time, second_side_time, batch_data_size = 0, 0, 0
    class_num = 10 # for cifar-10
    # w_glob_client_tier = []
    
    init_glob_model = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)
    
    # w_glob_client_tier = net_glob_client_tier.state_dict()
    
    
    for tier in range(0, num_tiers + 1):
        global net_glob_server_tier
        
        if tier == 0:
            net_glob_client_tier = resnet56_SFL_fedavg_base(classes=class_num,tier=1, fedavg_base = True)
        else:
            net_glob_client_tier, net_glob_server_tier = resnet56_SFL_local_tier_7(classes=class_num,tier=tier)
        
        
        net_glob_client_tier.train()
        w_glob_client_tier = net_glob_client_tier.state_dict()
        net_glob_client_tier.load_state_dict(w_glob_client_tier)
        
        
        first_side_time, second_side_time, batch_data_size = trainer(tier, net_glob_client_tier, net_glob_server_tier,
                                                                     w_glob_client_tier, dataset)
        print([first_side_time, second_side_time, batch_data_size])
        
        tier_sides_times_size.append([first_side_time, second_side_time, batch_data_size])
        
    return tier_sides_times_size

tier_profiler(128, 'cifar-10', num_tiers = 7)