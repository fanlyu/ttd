# -----------------------------------------------------------------------------
# Functions for model training
# -----------------------------------------------------------------------------
from model import get_ccd_model
from data import create_dataloader, create_dataloader_test
import torch

class Hash_Representation:
    def __init__(self, angle_num):
        self.topk = 3
        self.angle_num = angle_num
        self.hash_vectors = torch.randn(angle_num, 768).cpu()
        self.hash_vectors = self.hash_vectors / self.hash_vectors.norm(dim=1, keepdim=True)


    def map_to_hash(self, hash_value):
        hash = ''
        if hash_value[0] > 5.0:
            hash = hash + '50'
        elif hash_value[0] > 4.8 and hash_value[0] <= 5.0:
            hash = hash + '48'
        elif hash_value[0] > 4.6 and hash_value[0] <= 4.8:
            hash = hash + '46'
        elif hash_value[0] > 4.4 and hash_value[0] <= 4.6:
            hash = hash + '44'
        elif hash_value[0] > 4.2 and hash_value[0] <= 4.4:
            hash = hash + '42'
        elif hash_value[0] > 4.0 and hash_value[0] <= 4.2:
            hash = hash + '40'
        elif hash_value[0] > 3.8 and hash_value[0] <= 4.0:
            hash = hash + '38'
        elif hash_value[0] > 3.6 and hash_value[0] <= 3.8:
            hash = hash + '36'
        elif hash_value[0] > 3.4 and hash_value[0] <= 3.6:
            hash = hash + '34'
        elif hash_value[0] > 3.2 and hash_value[0] <= 3.4:
            hash = hash + '32'
        elif hash_value[0] > 3.0 and hash_value[0] <= 3.2:
            hash = hash + '30'
        elif hash_value[0] > 2.8 and hash_value[0] <= 3.0:
            hash = hash + '28'
        elif hash_value[0] > 2.6 and hash_value[0] <= 2.8:
            hash = hash + '26'
        elif hash_value[0] > 2.4 and hash_value[0] <= 2.6:
            hash = hash + '24'
        elif hash_value[0] > 2.2 and hash_value[0] <= 2.4:
            hash = hash + '22'
        elif hash_value[0] > 0 and hash_value[0] <= 2.2:
            hash = hash + '20'
        else:
            raise Exception("Error Hash Value")
        
        for a in hash_value[1:]:
            hash = hash + str(int(a))
        
        return hash
    
    def merge_dicts_with_samples_and_hashes(self, memory, memory_hash):
        new_dict = {}
        for label in memory:
            samples = memory[label]
            hashes = memory_hash[label]

            if len(samples) != len(hashes):
                raise ValueError(f"label {label} : sample and hash not match!")

            for sample, hash_value in zip(samples, hashes):
                if hash_value not in new_dict:
                    new_dict[hash_value] = []
                new_dict[hash_value].append((sample, label))
        
        return new_dict


class Trainer:
    def __init__(self, args, model, stage_i, dataset_train, dataset_val, dataset_test):
        self.args = args
        self.stage_i = stage_i
        self.ccd_model = get_ccd_model(self.args, model, self.stage_i)
        self.train_dataloader_i = create_dataloader(self.args, dataset_train[stage_i], self.stage_i) 
        self.val_dataloader = create_dataloader(self.args, dataset_val, -1)

        self.train_dataloader_i_test = create_dataloader_test(self.args, dataset_test[stage_i], self.stage_i)

        if self.args.transductive_evaluation:
            self.test_dataloader_i = create_dataloader(self.args, dataset_train[:self.stage_i + 1], -2)
        else:
            self.test_dataloader_i = create_dataloader(self.args, dataset_test[:self.stage_i + 1], -2)


    
    def run(self):

        HASH = Hash_Representation(angle_num=6)
        

        if self.args.train:
            if self.stage_i == 0:
                print("stage_i = 0,use train,fit")
                # model = self.ccd_model.fit(self.train_dataloader_i, self.val_dataloader)
                # model = self.ccd_model.fit_mag(self.train_dataloader_i, self.val_dataloader)
                model = self.ccd_model.fit_notrain(self.train_dataloader_i, self.val_dataloader, self.test_dataloader_i)

            if self.stage_i > 0:
                print("stage_i > 0,use test,pred_and_fit")
                # model = self.ccd_model.learnCentroid(self.train_dataloader_i)
                model = self.ccd_model.TTT(self.train_dataloader_i_test, HASH)
                
            return model
        

        if self.args.test and self.stage_i > 0:
            self.ccd_model.eval(self.test_dataloader_i)
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