from dataclasses import dataclass

import numpy as np
from numpy._typing import _array_like

from exceptions import NotTrainedError


@dataclass
class LinearRegression:
    
    weights: np.array | None = None
    _is_trained: bool = False


    def train(self,X: _array_like, y: _array_like):
        X_= np.c_[np.ones(len(X)), X]
        self.weights = np.linalg.pinv(X_)@y
        self._is_trained = True
        

    def predict(self,X: _array_like):
        if not self._is_trained:
            raise NotTrainedError()
        X_= np.c_[np.ones(len(X)), X]
        return X_@self.weights
        
