import numpy as np

class Network:
    
    
    # constructor, initialize matrixes
    def __init__(self, sizes):

        self.sizes = [1, 5, 1]
        self.learning_rate = 0.1
        self.hidden_layer_num = 1

        # forward propagation
        # input layer
        self.inputs = []
        # output layer
        self.expec_outputs = []
        self.outputs_in = []
        self.outputs_in_deri = []
        self.outputs_product = []
        self.outputs = []
        self.outputs_bias = []
        self.outputs_weight = []
        # hidden layers
        # input part of hidden layers
        self.input_layers = []
        # output part of hidden layers
        self.output_layers = []
        self.biases = []
        self.weights = []

        # back propagation
        # derivatives of threshold function
        self.thf_deris = []
        # derivatives of output part in hidden layers
        self.out_deris = []
        # hidden layers weights derivative
        self.weight_deris = []
        # product of thf_deris and out_deris: [a, b] * [c, d] = [ac, bd]
        self.thf_out_products = []
        self.bias_deris = []
        # outputs derivative
        self.outputs_deri = []
        # outputs weight derivative
        self.outputs_weight_deri = []
        # outputs bias derivative
        self.outputs_bias_deri = []

        if len(sizes) >= 3:
            self.sizes = sizes
        self.hidden_layer_num = len(self.sizes) - 2

        # initialize hidden layers
        for size in self.sizes[1: -1]:
            self.input_layers.append(np.zeros(size))
            self.output_layers.append(np.zeros(size))
            self.biases.append((np.ones(size)))
            self.thf_deris.append(np.zeros(size))
            self.out_deris.append(np.zeros(size))
            self.thf_out_products.append(np.zeros(size))
            self.bias_deris.append(np.zeros(size))
        
        # initialize weights of hidden layers
        for i in range(len(self.sizes) - 2):
            self.weights.append((np.random.rand(self.sizes[i], self.sizes[i + 1]) * 0.2 - 0.1))
            self.weight_deris.append(np.zeros((self.sizes[i], self.sizes[i + 1])))

        # initialize output layer
        self.outputs_weight = np.random.rand(self.sizes[-2], self.sizes[-1]) * 0.2 - 0.1
        self.outputs_bias = np.ones(self.sizes[-1])
        
    
    def __forward(self):        
        # compute the first layer
        self.input_layers[0] = np.dot(self.inputs, self.weights[0]) - self.biases[0]
        self.output_layers[0] = self.__thres_fun(self.input_layers[0])
        
        # loop for the other hidden layers
        for i in range(1, self.hidden_layer_num):
            self.input_layers[i] = np.dot(self.output_layers[i - 1], self.weights[i]) - self.biases[i]
            self.output_layers[i] = self.__thres_fun(self.input_layers[i])

        # compute the output layer
        self.outputs_in = np.dot(self.output_layers[-1], self.outputs_weight) - self.outputs_bias
        self.outputs = self.__sigmiodal(self.outputs_in)

    def __back(self):
        # derivatives of output layers
        self.outputs_deri = self.outputs - self.expec_outputs
        self.outputs_in_deri = self.__thres_fun_deri(self.outputs_in)
        self.outputs_product = self.outputs_in_deri * self.outputs_deri

        # first hidden layer from back
        self.out_deris[-1] = np.dot(self.outputs_weight, self.outputs_product)
        self.thf_deris[-1] = self.__thres_fun_deri(self.input_layers[-1])
        self.thf_out_products[-1] = self.thf_deris[-1] * self.out_deris[-1]
        
        # the other layers
        for i in range(len(self.output_layers) - 2, -1, -1):
            self.out_deris[i] = np.dot(self.weights[i + 1], self.thf_out_products[i + 1])
            self.thf_deris[i] = self.__thres_fun_deri(self.input_layers[i])
            self.thf_out_products[i] = self.thf_deris[i] * self.out_deris[i]

        # derivatives of weights
        # weight of output layer
        row = len(self.output_layers[-1])
        col = len(self.outputs_deri)
        self.outputs_weight_deri = np.dot(self.output_layers[-1].reshape(row, 1), 
                                    self.outputs_product.reshape(1, col))
        # weight of first layer
        row = len(self.inputs)
        col = len(self.thf_out_products[0])
        self.weight_deris[0] = np.dot(self.inputs.reshape(row, 1), 
                                self.thf_out_products[0].reshape(1, col))

        # the other layers
        for i in range(1, len(self.weight_deris)):
            row = len(self.output_layers[i - 1])
            col = len(self.thf_out_products[i])
            self.weight_deris[i] = np.dot(self.output_layers[i - 1].reshape(row, 1), 
                                    self.thf_out_products[i].reshape(1, col))
        
        # derivatives of hidden layer biases
        for i in range(len(self.bias_deris)):
            self.bias_deris[i] = self.thf_out_products[i] * (-1)
        # derivative of output layer bias
        self.outputs_bias_deri = self.outputs_product * (-1)

        # update weights and biases with learning rate
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.learning_rate * self.weight_deris[i]
            self.biases[i] = self.biases[i] - self.learning_rate * self.bias_deris[i]
        self.outputs_weight = self.outputs_weight - self.learning_rate * self.outputs_weight_deri
        self.outputs_bias = self.outputs_bias - self.learning_rate * self.outputs_bias_deri

    def __thres_fun(self, inputs):
        return self.__sigmiodal(inputs)
        # return self.__linear(inputs)

    def __thres_fun_deri(self, inputs):
        return self.__sigmiodal_deri(inputs)
        # return self.__linear_deri(inputs)
         
    def __sigmiodal(self, inputs):
        return (np.exp(inputs) / (1 + np.exp(inputs)))
    
    def __sigmiodal_deri(self, inputs):
        tmp = self.__sigmiodal(inputs)
        return tmp * (1 - tmp)
    
    def __linear(self, inputs):
        return np.array(inputs)
    
    def __linear_deri(self, inputs):
        return np.ones(inputs.shape)
        
    def __loss(self):
        loss_value = 0
        for i in range(len(self.expec_outputs)):
            loss_value += (self.expec_outputs[i] - self.outputs[i]) ** 2
        loss_value = loss_value * 0.5
        return loss_value

    def train(self, inputs, expec_outputs, learning_rate):
        self.inputs = np.array(inputs)
        self.expec_outputs = np.array(expec_outputs)
        self.learning_rate = learning_rate

        self.__forward()
        self.__back()
    
    def compute(self, inputs):
        self.inputs = np.array(inputs)
        self.__forward()
        return self.outputs