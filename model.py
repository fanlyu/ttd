# -----------------------------------------------------------------------------
# Functions for model configuration and training
# -----------------------------------------------------------------------------
import os

import util.globals as globals
from util.util import info
from models import get_model


class TTD_L2P_known_K_VIT_SSK_model:
    def __init__(self, args, model, stage_i):
        super().__init__()
        self.args = args
        self.stage_i = stage_i
        self.model = get_model(args)
        self.contrastive_model = self.model['ttd_model'](self.args, model, self.stage_i)

        if self.stage_i == 0:
            globals.discovered_K = self.args.labelled_data


    def fitobj(self, train_dataloader, val_dataloader):
        model = None
        model_path = os.path.join(self.args.save_path, 'model', f"{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt")
        if not self.args.train:
            info(f"Training process is not performed")
            info(f"{model_path} exists, go to eval")
            model = (self.contrastive_model.model, self.contrastive_model.original_model, self.contrastive_model.projection_head)
        else: 
            info(f"Start training process for {self.args.ttd_model}, stage {self.stage_i}")
            model = self.contrastive_model.fitqiu(train_dataloader, val_dataloader)
        
        return model


    def eval(self, test_dataloader):
        if self.args.test: 
            self.contrastive_model.eval(test_dataloader)
        else: 
            info(f"Evaluation process is not performed.")


    def learnCentroid(self, train_dataloader):
        model = None
        model_path = os.path.join(self.args.save_path, 'model', f"{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt")
        if not self.args.train:
            info(f"Training process is not performed")
            info(f"{model_path} exists, go to eval")
            model = (self.contrastive_model.model, self.contrastive_model.original_model, self.contrastive_model.projection_head)
        else: 
            info(f"Start training process for {self.args.ttd_model}, stage {self.stage_i}")
            model = self.contrastive_model.learnCentroid(train_dataloader)
        
        return model

    def fit_notrain(self, train_dataloader, val_dataloader, test_dataloader):
        model = None
        model_path = os.path.join(self.args.save_path, 'model', f"{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt")
        if not self.args.train:
            info(f"Training process is not performed")
            info(f"{model_path} exists, go to eval")
            model = (self.contrastive_model.model, self.contrastive_model.original_model, self.contrastive_model.projection_head)
        else: 
            info(f"{model_path} exists, go to eval")
            model = (self.contrastive_model.model, self.contrastive_model.original_model, self.contrastive_model.projection_head)

        return model
    
    def TTT(self, train_dataloader_i, HASH):
        model = None
        model_path = os.path.join(self.args.save_path, 'model', f"{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt")
        if not self.args.train:
            info(f"Training process is not performed")
            info(f"{model_path} exists, go to eval")
            model = (self.contrastive_model.model, self.contrastive_model.original_model, self.contrastive_model.projection_head)

        else: 
            info(f"Start training process for {self.args.ttd_model}, stage {self.stage_i}")
            model = self.contrastive_model.TTT(train_dataloader_i, HASH)
        
        return model

    


get_model_dict = {
    'TTD_L2P_known_K': TTD_L2P_known_K_VIT_SSK_model,
}


def get_ttd_model(args, trained_model, stage_i):
    '''
    Input: model parse 
    Return: lightning training module
    '''
    if stage_i != -1:
        model_parse = args.ttd_model
        model = get_model_dict[model_parse](args, trained_model, stage_i)

    else:
        model_parse = args.ttd_model
        model = get_model_dict[model_parse](args, trained_model, stage_i)
    
    if model == None:
        raise NotImplementedError(f"Model --> {model_parse} is not implemented")
    return model

    