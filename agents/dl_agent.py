from model.agent import Agent
from graphs.models import *
from data_loader import *

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch
import os



class DeepLearningAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        # save important parameter
        self.max_epochs = config['max_epochs'] if 'max_epochs' in config else 1
        self.verbose = True if 'report_freq' in config else False
        self.report_freq = config['report_freq'] if 'report_freq' in config else 1
        self.validate_every = config['validate_every'] if 'validate_every' in config else self.max_epochs

        
        self.criterion = getattr(nn, config['criterion'], None)(**config['criterion_args'])

        self.model = globals()[config['model']](**config['model_args'])
        self.parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = getattr(optim, config['optimizer'], None)(self.parameters,
                                                                    **config['optimizer_args'])

        if 'scheduler' in config:
            self.optimizer = getattr(scheduler, config['scheduler'])(self.optimizer, self.max_epochs)
    

        if 'data_loader' in config and 'data_loader_args' in config:
            data_loader = globals()[config['data_loader']](**config['data_loader_args'])
            self.train_queue = data_loader.train_loader
            self.valid_queue = data_loader.test_loader

        # initialize counter
        self.current_epoch = 1
        self.current_iter = 1

        # set cuda flag
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda and not self.config['cuda']:
            print("WARNING: You have a CUDA device, so you should enable CUDA")

        self.cuda = self.has_cuda and self.config['cuda']

        # get device
        self.device = device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        print("Program will run on *****{}*****".format(self.device))

        # set manual seed
        self.manual_seed = config['seed']
        torch.manual_seed(self.manual_seed)
        # load checkpoint
        # self.load_checkpoint(self.config.checkpoint_file)

        # summary writer
        self.summary_writer = SummaryWriter() if 'summary_writer' in config and config['summary_writer'] else None

        # save path
        self.save_path = './pretrained_weights'

        # default messages
        self.validate_msg = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
        self.train_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'

        self.train_err = self.train_loss = self.test_err = self.test_loss = 0


    def run(self):
        try:
            if self.config['mode'] == 'train':
                self.to(self.device)
                self.train()
            elif self.config['mode'] == 'eval':
                self.validate()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize") 

    def train(self):
        while self.current_epoch < self.max_epochs:
            self.train_one_epoch()
            if self.current_epoch % self.validate_every == 0:
                self.validate()
            self.current_epoch += 1


    def train_one_epoch(self):
        self.model.train()
        correct = total = train_loss = 0
        n_inputs = len(self.train_queue.dataset)
        for step, (inputs, targets) in enumerate(self.train_queue):
            targets, loss, outputs = self.feed_forward(inputs, targets)

            train_loss += loss.item()
            predicted = self.predict(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            
            if self.verbose and step % self.report_freq == 0:
                percentage = 100.*total/n_inputs
                print(self.train_msg.format(self.current_epoch, total, n_inputs, percentage, loss.item()))

            self.current_iter += 1
        
        avg_loss = train_loss/total
        err = 100.*(1- (correct/total))
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Accuracy/train', err, self.current_epoch)

        # torch.cuda.empty_cache()
        return err, avg_loss

    def feed_forward(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return targets, loss, outputs

    def validate(self):
        self.model.eval()
        test_loss = correct = total = 0

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.valid_queue):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item()  # sum up batch loss

                predicted = self.predict(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets.view_as(predicted)).sum().item()

                if self.verbose and step % self.report_freq == 0:
                    avg_loss = test_loss/total
                    acc = 100.*correct/total
                    print(self.validate_msg.format(avg_loss, correct, total, acc))

        avg_loss = test_loss/total
        err = 100.*(1- (correct/total))
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/test', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Accuracy/test', err, self.current_epoch)

        return err, avg_loss

    def predict(self, outputs):
        if outputs.shape[1] == 1:
            pred = outputs
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            return pred
        return outputs.max(dim=1, keepdims=True)[1]

    def finalize(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth.tar'))

        

