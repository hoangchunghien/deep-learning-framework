import numpy as np

class Tensor(object):
    def __init__(self, data, parents=None, op=None):
        """
        - parents: A list of Tensors used in creation of current Tensor
        - op: The computation operator used in creation of current Tensor
        """
        self.data = np.array(data)
        self.op = op
        self.parents = parents
        self.grad = None
    
    def backward(self, grad):
        self.grad = grad

        if self.op == 'add':
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)

    def __add__(self, other):
        return Tensor(self.data + other.data, parents=[self, other], op="add")
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
