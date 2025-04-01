## Lets's try to use some NN
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers):
        super(MLP, self).__init__()
        initial_dim=input_size
        layers=[]
        for i in range(num_layers):
            layers.append(nn.Linear(initial_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            initial_dim=hidden_size
        layers.append(nn.Linear(initial_dim, output_size))
        layers.append(nn.Sigmoid())
        self.layers=nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)
    

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.input_size=input_size
        self.conv1=nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.fc1=nn.Linear(hidden_size, output_size)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0), -1)
        x=self.fc1(x)
        return self.sigmoid(x)
    


class TorchToEstimator(BaseEstimator):
    def __init__(self,input_size,hidden_size,output_size,num_layers,criterion,optimizer,num_epochs):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers
        if criterion is None:
            self.criterion=nn.MSELoss()
        else:
            self.criterion=criterion
        if optimizer is None:
            self.optimizer=optim.Adam
        else:
            self.optimizer=optimizer
        self.output_size=1
        self.num_epochs=num_epochs
        self.model=MLP(input_size,hidden_size,output_size,num_layers)
        self.trained=False
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer(self.model.parameters(),lr=0.0001)


    def _train(self,X:pd.DataFrame,y:pd.Series,verbose=False):
        X=torch.tensor(X.values,dtype=torch.float32)
        y=torch.tensor(y.values,dtype=torch.float32)
        dataset=torch.utils.data.TensorDataset(X,y)
        dataloader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=True)
        optimizer=self.optimizer(self.model.parameters(),lr=0.0001)
        self.model.train()
        for epoch in range(self.num_epochs):
            for inputs,labels in dataloader:
                optimizer.zero_grad()
                outputs=self.model(inputs.to(self.device))
                loss=self.criterion(outputs.view(-1),labels.view(-1).to(self.device))
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 and verbose:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
        self.trained=True
    
    def fit(self,X,y):
        self._train(X,y)
        return self
    
    def predict(self,X):
        X=torch.tensor(X.values,dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs=self.model(X.to(self.device))
            return np.round(outputs.cpu().numpy()).reshape(-1)
        
    def score(self,X,y):
        y_pred=self.predict(X)
        return np.sum(y_pred==y)/len(y)