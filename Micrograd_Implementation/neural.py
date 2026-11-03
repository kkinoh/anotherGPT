import random
from Micrograd_Implementation.engine import Value
"""
Module -> Neuron -> Layer -> MLP.
"""
class Module:
    def set_grad_to_zero(self):
        for param in self.parameters():
            param.grad = 0
        return

    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, num_of_inputs, is_nonlinear=True):
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(num_of_inputs)] # [Value]
        self.bias = Value(0)
        self.is_nonlinear = is_nonlinear
        return

    def __call__(self, x): # When called, returns Value for this Neuron.
        weights_and_xs = zip(self.weight, x)
        result = sum((wi * xi for wi, xi in weights_and_xs), self.bias)
        # Note that result is a Value, because wi's and self.bias are Values.
        return result.relu() if self.is_nonlinear else result

    def parameters(self):
        return self.weight + [self.bias] # Parameters should be Values.

    def __repr__(self):
        return f'{'ReLU' if self.is_nonlinear else 'Linear'} Neuron with {len(self.weight)} inputs.'

class Layer(Module):
    def __init__(self, num_of_inputs, num_of_outputs, is_nonlinear=True):
        self.neurons = [Neuron(num_of_inputs, is_nonlinear) for _ in range(num_of_outputs)] # A list of Neurons.
        self.inputs = num_of_inputs
        self.outputs = num_of_outputs
        self.is_nonlinear = is_nonlinear
        return

    def __call__(self, layer_values): # When called, returns a list with length of the number of neurons.
        result = [self.neurons[i](layer_values) for i in range(self.outputs)]
        # for i in range(self.outputs):
        #    result.append(self.neurons[i](layer_values))
            # Every iteration adds a neuron-output-Value to the result (output list).
        return result

    def parameters(self):
        result = []
        for neu in self.neurons:
            for param in neu.parameters():
                result.append(param)

        return result # Elements in result must be Values, such that their grad can be set to zero.

    def __repr__(self):
        return f'Layer of {len(self.neurons)} Neurons, including {self.parameters()}'

class MLP(Module):
    def __init__(self, sizes, is_nonlinear=True):
        self.sizes = sizes # Sizes are number of elements in input vector, layers, and output vector.
        self.is_nonlinear = is_nonlinear
        self.layer = []
        len_of_size = len(self.sizes)
        for i in range(len_of_size - 2):
            self.layer.append(Layer(self.sizes[i], self.sizes[i + 1], self.is_nonlinear))

        self.layer.append(Layer(self.sizes[len_of_size - 2], self.sizes[len_of_size - 1], False))
        """Note that last Layer must not use ReLU, or else the MLP return value may be zero."""

        return
    
    def __call__(self, input_vector): # When called, call every Layer in turn.
        hidden_vector = input_vector
        for i in self.layer:
            hidden_vector = i(hidden_vector)

        return hidden_vector # This is the score. Might be 1-D, might not be.

    def parameters(self):
        result = []
        for lay in self.layer:
            for param in lay.parameters():
                result.append(param)

        return result # Same as Layer, elements in result must be Values.
