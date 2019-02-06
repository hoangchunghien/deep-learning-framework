
from nn import Layer

class MSELoss(Layer):
    def __init__(self):
        super().__init__()
  
    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).mean(0)

class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return input.cross_entropy(target)
