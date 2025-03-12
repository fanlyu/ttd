# -----------------------------------------------------------------------------
# Functions for model training
# -----------------------------------------------------------------------------
from model import get_ttd_model
from data import create_dataloader, create_dataloader_test
from tqdm import tqdm
import torch


import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE

from hashmemoryTTD import Hash_Representation



class Trainer:
    '''
    To construct instance of trainer object for each stage
    '''
    def __init__(self, args, model, stage_i, dataset_train, dataset_val, dataset_test):
        self.args = args
        self.stage_i = stage_i
        self.ttd_model = get_ttd_model(self.args, model, self.stage_i)
        self.train_dataloader_i = create_dataloader(self.args, dataset_train[stage_i], self.stage_i) 
        self.val_dataloader = create_dataloader(self.args, dataset_val, -1)

        self.train_dataloader_i_test = create_dataloader_test(self.args, dataset_test[stage_i], self.stage_i)
        
        if self.args.transductive_evaluation:
            self.test_dataloader_i = create_dataloader(self.args, dataset_train[:self.stage_i + 1], -2)
        else:
            self.test_dataloader_i = create_dataloader(self.args, dataset_test[:self.stage_i + 1], -2)


    
    def run(self):
        HASH = Hash_Representation( angle_num=6)
        
        if self.args.train:
            if self.stage_i == 0:
                print("stage_i = 0,use train,fit")
                # model = self.ttd_model.fitobj(self.train_dataloader_i, self.val_dataloader)
                model = self.ttd_model.fit_notrain(self.train_dataloader_i, self.val_dataloader, self.test_dataloader_i)

            if self.stage_i > 0:
                print("stage_i > 0,use test,pred_and_fit")
                # model = self.ttd_model.learnCentroid(self.train_dataloader_i)
                model = self.ttd_model.TTT(self.train_dataloader_i_test, HASH)

            return model
        

        if self.args.test and self.stage_i > 0:
            self.ttd_model.eval(self.test_dataloader_i)
            return None
        

    



def RunContinualTrainer(args, datasets_train, datasets_val, datasets_test):
    model = None

    for stage_i in range(args.n_stage+1):
        model = Trainer(
            args, 
            model, 
            stage_i, 
            datasets_train, 
            datasets_val, 
            datasets_test,
        ).run()