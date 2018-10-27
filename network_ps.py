import numpy as np 

class Network:
    
    def __init__(self, sizes):
        self.para = {
            'learning_rate': 0.1,
            'sizes': sizes,
            'hl_num': len(sizes) - 2,
            
            # forward propagation
            # inputs and outputs layer
            'inputs': [],
            'expec_outputs': [],
            'outputs': [],
            'outputs_bias': [],
            'outputs_weight': [],

            # hidden layer
            'hl_inputs': [],
            'hl_outputs': [],
            'weights': [],
            'biases': [],

            # back propagation
            # derivatives of threshold function
            'thf_deris': [],
            # derivatives of hidden layers outputs
            'hl_out_deris': [],
            # derivatives of weights
            'weight_deris': [],
            # product of thf_deris and hl_out_deris: [a, b] * [c, d] = [ac, bd]
            'thf_out_products': [],
            # derivatives of biases
            'bias_deris':[],
            # outputs derivatives
            'outputs_deris': [],
            # outputs weights derivates
            'outputs_weight_deris': [],
            # outputs bias derivatives
            'outputs_bias_deris': []
        }

class Network_ps(Network):
    
    # constructor, initialize matrixes
    def __init__(self, sizes):
        Network.__init__(self, sizes)

        # initialize hidden layers
        for size in self.para['sizes'][1:-1]:
            # initialize hidden layer property with 0
            self.para['hl_inputs'].append(np.zeros(size))
            self.para['hl_outputs'].append(np.zeros(size))
            self.para['thf_deris'].append(np.zeros(size))
            self.para['hl_out_deris'].append(np.zeros(size))
            self.para['thf_out_product'].append(np.zeros(size))
            self.para['bias_deris'].append(np.zeros(size))

            # initialize bias with value -0.5
            self.para['biases'].append(np.zeros(size)-0.5)
        
        # initialize weights of hidden layers
        for i in range(len(self.para['sizes']) - 2):
            self.para['weights'].append((np.random.rand(self.para['sizes'][i], self.para['sizes'][i+1]) * 2 - 1))
            self.para['weight_deris'].append(np.zeros((self.para['sizes'][i], self.para['sizes'][i+1])))

        # initialize output layer
        self.para['outputs_weight'] = np.random.rand(self.para['sizes'][-2], self.para['sizes'][-1]) * 2 - 1
        self.para['outputs_bias'] = np.zeros(self.para['sizes'][-1]) - 0.5

class Network_ws(Network):

    def __init__(self, sizes):
        Network.__init__(self, sizes)

        # initialize hidden layers with None
        for size in self.para['sizes'][1:-1]:
            self.para['hl_inputs'].append(None)
            self.para['hl_outputs'].append(None)
            self.para['thf_deris'].append(None)
            self.para['hl_out_deris'].append(None)
            self.para['thf_out_product'].append(None)
            self.para['bias_deris'].append(None)
            self.para['biases'].append(None)
        
        # initialize weights of hidden layers with None
        for i in range(len(self.para['sizes']) - 2):
            self.para['weights'].append(None)
            self.para['weight_deris'].append(None)