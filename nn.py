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
                (all([not cnt != 0 for _, cnt in self.children.items()]) or grad_origin is None):
                # Check whether a tensor receive enough number of gradients from each child
                
                if self.op == 'add':
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad, self)
                
                if self.op == "neg":
                    self.parents[0].backward(self.grad.__neg__())
                
                if self.op == "sub":
                    new = Tensor(self.grad.data)
                    self.parents[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.parents[1].backward(new, self)
                
                if self.op == "mul":
                    new = self.grad * self.parents[1]
                    self.parents[0].backward(new, self)
                    new = self.grad * self.parents[0]
                    self.parents[1].backward(new, self)
                
                if self.op == "mm":
                    new = self.grad.mm(self.parents[1].transpose())
                    self.parents[0].backward(new)
                    new = self.grad.transpose().mm(self.parents[0]).transpose()
                    self.parents[1].backward(new)
                
                if self.op == "transpose":
                    self.parents[0].backward(self.grad.transpose())
                
                if "sum" in self.op:
                    axis = int(self.op.split("_")[1])
                    ds = self.parents[0].data.shape[axis]
                    self.parents[0].backward(self.grad.expand(axis, ds))
                
                if "mean" in self.op:
                    axis = int(self.op.split("_")[1])
                    ds = self.parents[0].data.shape[axis]
                    self.parents[0].backward(self.grad.expand(axis, ds))
                
                if "expand" in self.op:
                    axis = int(self.op.split("_")[1])
                    self.parents[0].backward(self.grad.sum(axis))
                
                if self.op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (self * (ones - self)))
                
                if self.op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (ones - (self * self)))
                
                if self.op == "index_select":
                    new_grad = np.zeros_like(self.parents[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.parents[0].backward(Tensor(new_grad))
                
                if self.op == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.parents[0].backward(Tensor(dx))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, parents=[self, other], op="add")
        return Tensor(self.data + other.data)
    
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, parents=[self], op="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        if self.autograd:
            return Tensor(self.data - other.data, autograd=True, parents=[self, other], op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if self.autograd:
            return Tensor(self.data * other.data, autograd=True, parents=[self, other], op="mul")
        return Tensor(self.data * other.data)
    
    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data), autograd=True, parents=[self, x], op="mm")
        return Tensor(self.data.dot(x.data))
    
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, parents=[self], op="transpose")
        return Tensor(self.data.transpose())
    
    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), autograd=True, parents=[self], op="sum_"+str(axis))
        return Tensor(self.data.sum(axis))
    
    def mean(self, axis):
        if self.autograd:
            return Tensor(self.data.mean(axis), autograd=True, parents=[self], op="mean_"+str(axis))
        return Tensor(self.data.mean(axis))
    
    def expand(self, axis, copies):
        transpose_cmd = list(range(0, len(self.data.shape)))
        transpose_cmd.insert(axis, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(transpose_cmd)
        
        if self.autograd:
            return Tensor(new_data, autograd=True, parents=[self], op="expand_"+str(axis))
        return Tensor(new_data)
    
    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1+np.exp(-self.data)), autograd=True, parents=[self], op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, parents=[self], op="tanh")
        return Tensor(np.tanh(self.data))
    
    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data], autograd=True, parents=[self], op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])
    
    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_ = temp / np.sum(temp, axis=len(self.data.shape)-1, keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if (self.autograd):
            out = Tensor(loss, autograd=True, parents=[self], op="cross_entropy")
            out.softmax_output = softmax_
            out.target_dist = target_dist
            return out
        return Tensor(loss)
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())


class Layer(object):
    def __init__(self):
        self.parameters = list()
    
    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        w = np.random.randn(n_inputs, n_outputs) * 0.1
        self.weight = Tensor(w, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)
    
    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        weight = np.random.rand(vocab_size, dim) / dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)
    
    def forward(self, input):
        return self.weight.index_select(input)


class RNNCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output, activation="sigmoid"):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise Exception("Unsupported non-linearity")
        
        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()
    
    def forward(self, input, hidden):
        prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden
    
    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
