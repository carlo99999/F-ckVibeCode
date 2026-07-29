from typing import Tuple
from dataclasses import dataclass

import numpy as np
from numpy._typing import _array_like
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from exceptions import NotTrainedError


class LinearRegression:

    def __init(self):
       self.weights: np.array | None = None
       self._is_trained: bool = False


    def train(self,X: _array_like, y: _array_like):
        X_= np.c_[np.ones(len(X)), X]
        self.weights = np.linalg.pinv(X_)@y
        self._is_trained = True
        

    def predict(self,X: _array_like):
        if not self._is_trained:
            raise NotTrainedError()
        X_= np.c_[np.ones(len(X)), X]
        return X_@self.weights
        


def load_dataset()->pd.DataFrame:
    return pd.read_csv('dataset_linear_regression.csv')


def train_test_loading(test_split:float=0.2)->tuple[pd.DataFrame,pd.DataFrame]:
    df = load_dataset()
    df_train,df_test = train_test_split(df,test_size=test_split)
    return df_train,df_test

def separate_x_y(df:pd.DataFrame,target_name:str)->tuple[pd.DataFrame,pd.Series]:
    return df.drop(columns=[target_name]), df[target_name]


def create_training():
    target_col='punteggio_esame'
    df_train,df_test = train_test_loading()
    X_train,y_train= separate_x_y(df_train,target_col)
    X_test,y_test= separate_x_y(df_test,target_col)

    X_train = X_train.to_numpy()
    X_test= X_test.to_numpy()

    linear_regression = LinearRegression()
    linear_regression.train(X_train,y_train)

    preds= linear_regression.predict(X_test)

    mse = mean_squared_error(y_test,preds)

    print(mse)
    

if __name__=="__main__":
    create_training()
    
    