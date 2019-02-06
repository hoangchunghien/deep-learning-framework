import numpy as np

class Tensor(object):
    def __init__(self, data, autograd=False, parents=None, op=None, id=None):
        """
        - parents: A list of Tensors used in creation of current Tensor
        - op: The computation operator used in creation of current Tensor
        """
        self.data = np.array(data)
        self.op = op
        self.parents = parents
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            id = np.random.randint(0, 1000000)
        self.id = id

        if self.parents is not None:
            for p in self.parents:
                # Keep track how many children a Tensor has
                if self.id not in p.children:
                    p.children[self.id] = 1
                else:
                    p.children[self.id] += 1
    
    def backward(self, grad, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                # Check to make sure tensor can perform backprops 
                # or wait for the last child update it grad
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
            
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad  # Accumulates grad from many children

            if self.parents is not None and \
                # Check whether a tensor receive enough number of gradients from each child
                (all([not cnt != 0 for _, cnt in self.children.items()]) or grad_origin is None):
                
                if self.op == 'add':
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad, self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, parents=[self, other], op="add")
        return Tensor(self.data + other.data)
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
