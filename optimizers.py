
class SGD(object):
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr
    
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0
    
    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.lr

        if zero:
            p.grad.data *= 0
