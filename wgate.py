from lasagne import init
from lasagne import nonlinearities


class WGate(object):
    def __init__(self, W_x=init.Normal(1.),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_x = W_x
        self.b = b
        self.nonlinearity = nonlinearity

class HGate(object):
    def __init__(self, W_c=init.Normal(1.),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_c = W_c
        self.b = b
        self.nonlinearity = nonlinearity