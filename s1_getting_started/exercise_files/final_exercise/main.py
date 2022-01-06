import argparse
import time
import sys
import os
from numpy import mod
import torch
from torch import nn, optim
from data import mnist
from model import MyAwesomeModel
import pdb
import numpy as np
import matplotlib.pyplot as plt

#from s1_getting_started.exercise_files.fc_model import train


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either trainingtrain or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        learning_rate = args.lr
        experiment_time = time.strftime("%Y%m%d-%H%M%S")
        model_name = experiment_time + '_MyAwesomeModel' + '.pt'
       
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_set, _ = mnist()
        # Extract images and labels form the loaded dataset
        img = torch.Tensor(train_set[train_set.files[0]])
        lbs = torch.Tensor(train_set[train_set.files[1]]).type(torch.LongTensor)
        # Convert images and labels into tensor and create dataloader
        train_dataset = torch.utils.data.TensorDataset(img, lbs)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle=True)

        epochs = 50
        training_loss = []

        for e in range(epochs):
            epoch_loss = 0
            model.train()
            for images, labels in trainloader:
                
                optimizer.zero_grad()
                outputs = model(images)
                # pdb.set_trace()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            training_loss.append(epoch_loss/len(trainloader))
            
            if e % 5 == 0:
                print(f'Epoch {e}, training loss: {training_loss[-1]}')

        # Plotting trainig curve
        epoch = np.arange(len(training_loss))
        plt.figure()
        plt.plot(epoch, training_loss, 'b', label='Training loss',)
        plt.legend()
        plt.xlabel('Epoch'), plt.ylabel('MSE Loss')
        plt.show()

        # Saving Model
        torch.save(model.state_dict(), os.path.join(os.getcwd(),'trained_models',model_name))
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        # print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(os.path.join(os.getcwd(),'trained_models',args.load_model_from))
        model.load_state_dict(state_dict)
        model.eval()

        criterion = nn.NLLLoss()
        
        _, test_set = mnist()
        # Extract images and labels form the loaded dataset
        img = torch.Tensor(test_set[test_set.files[0]])
        lbs = torch.Tensor(test_set[test_set.files[1]]).type(torch.LongTensor)
        # Convert images and labels into tensor and create dataloader
        test_dataset = torch.utils.data.TensorDataset(img, lbs)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle=True)

        with torch.no_grad():
            accuracy = 0
            test_loss = 0
            for images, labels in testloader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Accuracy
                ps = torch.exp(outputs)
                equality = (labels.data == ps.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print('Accuracy: {:2f}%'.format((accuracy/len(testloader))*100))

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    