class Value:
    """This is a Value Class. It constructs how values operate with each other."""
    def __init__(self, value, _children=(), _operation=''): # Note that _children and _operation are set to empty by default.
        self.data = value # The value of instance.
        self.previous = set(_children) # Previous values (instances) that make up this instance.
        self.operation = _operation # How this instance is made up.

        self.grad = 0 # The gradient (derivative dtop/dself).
        self._backward = lambda : None # How to calculate this gradient (derivative).
        return

    def __repr__(self):
        return f"Value of {self.data}, with gradient of {self.grad}.\n"

    """
    Below defines certain mathematical operations to be invoked upon Value.
    Backward calculates self.grad.
    """
    def relu(self):
        result = Value(0, (self,), 'ReLU') if self.data <= 0 else Value(self.data, (self,), 'ReLU')

        def _backward():
            self.grad += 0 if self.data <= 0 else result.grad
            return
        result._backward = _backward

        return result

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += result.grad
            other.grad += result.grad
            return
        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
            return
        result._backward = _backward

        return result

    def __pow__(self, other): # Other should not be Value instance.
        assert isinstance(other, int) or isinstance(other, float), 'Exponent should be int or float.'
        result = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad
            return
        result._backward = _backward

        return result

    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + other * (-1)

    def __rsub__(self, other):
        return other + self * (-1)
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * (other ** (-1))

    def __rtruediv__(self, other):
        return other * (self ** (-1))

    """Back propogation starts here."""
    def backward(self):
        topological_values = []
        calculated = set()
        """
        The build function assembles values in an order in which they are calculated.
        From lower ones to top ones.
        """
        def build(v: Value):
            if v not in calculated:
                calculated.add(v)
                for child in v.previous:
                    build(child)
                topological_values.append(v)
            return
        
        build(self) # Now a topological list of Values is constructed in topological_values.
        topological_values = reversed(topological_values) # This reverses the list to "from top to lower".
        # Note after debugging: Build must be before reversed.
        
        self.grad = 1.0
        for v in topological_values:
            v._backward()

        return
    