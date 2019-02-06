import numpy as np

class Tensor(object):
    def __init__(self, data):
        self.data = np.array(data)
    
    def __add__(self, other):
        return Tensor(self.data + other.data)
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
