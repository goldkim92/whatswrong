import os
import numpy as np 
from os.path import join
from tqdm import tqdm, tqdm_notebook

import torch
import torch.nn as nn

import dataloader_train
import dataloader_valid
import util

class Run_Imagenet():
    def __init__(self, args):
        self.phase = args.phase
        self.freeze = args.freeze
        self.model = args.model
        self.lr = args.lr
        self.bs = args.bs
        self.epochs = args.epochs
        self.log_step = args.log_step
        self.img_step = args.img_step
        self.img_dir = args.img_dir
        self.save_path = args.save_path
        self.load_path = args.load_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # load model
        if self.load_path is not None:
            self.load_model()    
        self.model.freeze_params(self.freeze)
        self.model = self.model.to(self.device)
#         self.model = nn.DataParallel(self.model)
        
        # build model
        self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
#                                          momentum=0.9,
#                                          weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.load_dataset()

        # image for checking activation features
        self.check_input = iter(self.valid_loader).next()[0][2:3]
        self.check_input = self.check_input.to(self.device)
        
    def load_dataset(self):
        if self.phase == 'train':
            self.train_loader = dataloader_train.imagenet_loader(self.bs)
        self.valid_loader = dataloader_valid.imagenet_loader(self.bs)
        
    def train(self):
#         lowest_loss = np.inf if self.load_path is None else self.lowest_loss
        lowest_loss = np.inf
        for epoch in range(self.epochs):
            print(f'===> EPOCH: {(epoch+1)}')
            for idx, (inputs, targets) in tqdm(enumerate(self.train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.model.train()
                outputs = self.model(inputs)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if (idx+1) % self.img_step == 0:
                    fname = join(self.img_dir,f'{epoch+1:02d}_{idx+1:04d}.png')
                    util.show_activation_features(self.check_input, self.model, fname)

                if (idx+1) % self.log_step == 0:
                    print(f'idx: {idx+1}')
                    current_loss = self.test(stop_idx=100)
                    if current_loss < lowest_loss:
                        lowest_loss = current_loss
                        self.save_model(current_loss)
                    
                        
    def test(self, stop_idx=None):
        top1_accuracy, top5_accuracy, avg_loss = 0., 0., 0.
        count = 0
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, targets) in tqdm(enumerate(self.valid_loader)):

                count += inputs.size(0)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets).detach().cpu().item()

                topk = outputs.topk(5,dim=1)[1]
                top1_accuracy += topk[:,0].eq(targets).sum().cpu().item()
                top5_accuracy += topk.eq(torch.stack([targets]*5,dim=1)).max(1)[0].sum().cpu().item()
                avg_loss += loss * inputs.size(0)

                torch.cuda.empty_cache()

                if stop_idx is not None:
                    if (idx+1) == stop_idx:
                        break
                    
        top1_accuracy /= count
        top5_accuracy /= count
        avg_loss /= count

        print('Classification')
        print(f'===> Test Loss: {avg_loss:.4f}, Top1-Acc: {top1_accuracy*100:.4f}, Top5-Acc: {top5_accuracy*100:.4f}')

        return avg_loss
    
    def save_model(self, loss):
        save_dict = {
            'valid_loss': loss,
            'model_state_dict': self.model.state_dict()
        }
        torch.save(save_dict, self.save_path)
        print('!!!! SAVE MODEL !!!!\n')
        
    def load_model(self):
        checkpoint = torch.load(self.load_path)
        self.lowest_loss = checkpoint['valid_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'!!!! LOAD MODEL, current loss: {self.lowest_loss:.04f} !!!!\n')
