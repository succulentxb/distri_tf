import numpy as np 
import json

LARGEST_RECV = 2**20

class Network:
    
    def __init__(self, sizes, learning_rate):
        self.para = {
            'learning_rate': learning_rate,
            'sizes': sizes,
            'hl_num': len(sizes) - 2,
            
            # forward propagation
            # inputs and outputs layer
            'inputs': [],
            'expec_outputs': [],
            'outputs': [],
            'outputs_bias': [],
            'outputs_weight': [],
            'outputs_in': [],

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
            'outputs_bias_deris': [],
            'outputs_in_deris': [],
            'outputs_products': []
        }

        for size in self.para['sizes'][1:-1]:
            # initialize hidden layer property with 0
            self.para['hl_inputs'].append(None)
            self.para['hl_outputs'].append(None)
            self.para['thf_deris'].append(None)
            self.para['hl_out_deris'].append(None)
            self.para['thf_out_products'].append(None)
            self.para['bias_deris'].append(None)

            # initialize bias with value -0.5
            self.para['biases'].append((np.zeros(size)-0.5).tolist())
        
        # initialize weights of hidden layers
        for i in range(len(self.para['sizes']) - 2):
            self.para['weights'].append((np.random.rand(self.para['sizes'][i], self.para['sizes'][i+1]) * 2 - 1).tolist())
            self.para['weight_deris'].append(None)

        # initialize output layer
        self.para['outputs_weight'] = (np.random.rand(self.para['sizes'][-2], self.para['sizes'][-1]) * 2 - 1).tolist()
        self.para['outputs_bias'] = (np.zeros(self.para['sizes'][-1]) - 0.5).tolist()
 

def sigmiodal_deri(inputs):
    tmp = np.exp(inputs) / (np.exp(inputs) + 1)
    return (tmp * (1-tmp)).tolist()

def back(network):
    network.para['outputs_deris'] = (np.array(network.para['outputs']) - network.para['expec_outputs']).tolist()
    network.para['outputs_in_deris'] = sigmiodal_deri(network.para['outputs_in'])
    network.para['outputs_products'] = (np.array(network.para['outputs_in_deris']) * network.para['outputs_deris']).tolist()

    network.para['hl_out_deris'][-1] = (np.dot(network.para['outputs_weight'], network.para['outputs_products'])).tolist()
    network.para['thf_deris'][-1] = sigmiodal_deri(network.para['hl_inputs'][-1])
    network.para['thf_out_products'][-1] = (np.array(network.para['thf_deris'][-1]) * network.para['hl_out_deris'][-1]).tolist()

    for i in range(len(network.para['hl_outputs'])-2, -1, -1):
        network.para['hl_out_deris'][i] = (np.dot(network.para['weights'][i+1], network.para['thf_out_products'][i+1])).tolist()
        network.para['thf_deris'][i] = sigmiodal_deri(network.para['hl_inputs'][i])
        network.para['thf_out_products'][i] = (np.array(network.para['thf_deris'][i]) * network.para['hl_out_deris'][i]).tolist()

    # compute weight derivatives of output layer
    row = len(network.para['hl_outputs'][-1])
    col = len(network.para['outputs_products'])
    network.para['outputs_weight_deris'] = (np.dot(np.array(network.para['hl_outputs'][-1]).reshape(row, 1), 
                                            np.array(network.para['outputs_products']).reshape(1, col))).tolist()
        
    # compute weight derivatives of input layer
    row = len(network.para['inputs'])
    col = len(network.para['thf_out_products'][0])
    network.para['weight_deris'][0] = (np.dot(np.array(network.para['inputs']).reshape(row, 1), 
                                        np.array(network.para['thf_out_products'][0]).reshape(1, col))).tolist()
        
    # weight derivatives of other layers
    for i in range(1, len(network.para['weight_deris'])):
        row = len(network.para['hl_outputs'][i-1])
        col = len(network.para['thf_out_products'][i])
        network.para['weight_deris'][i] = (np.dot(np.array(network.para['hl_outputs'][i-1]).reshape(row, 1), 
                                            np.array(network.para['thf_out_products'][i]).reshape(1, col))).tolist()

        # derivatives of hidden layer biases
    for i in range(len(network.para['bias_deris'])):
        network.para['bias_deris'][i] = (np.array(network.para['thf_out_products'][i]) * (-1)).tolist()
    # derivatives of output layer bias
    network.para['outputs_bias_deris'] = (np.array(network.para['outputs_products']) * (-1)).tolist()

        # update weights and biases with derivatives and learning rate
    for i in range(len(network.para['weights'])):
        network.para['weights'][i] = (np.array(network.para['weights'][i]) - 
                                    network.para['learning_rate'] * np.array(network.para['weight_deris'][i])).tolist()
        network.para['biases'][i] = (np.array(network.para['biases'][i]) - 
                                    network.para['learning_rate'] * np.array(network.para['bias_deris'][i])).tolist()

    network.para['outputs_weight'] = (np.array(network.para['outputs_weight']) - 
                                    network.para['learning_rate'] * np.array(network.para['outputs_weight_deris'])).tolist()
    network.para['outputs_bias'] = (np.array(network.para['outputs_bias']) - 
                                    network.para['learning_rate'] * np.array(network.para['outputs_bias_deris'])).tolist()

def tackle_ws_request(network, ws_ps_sock):
    '''
    tackle the requests from ws

    :para ws_ps_sock: socket between ws and ps
    :type ws_ps_sock: socket.socket
    '''
    while True:
        request = ws_ps_sock.recv(LARGEST_RECV)
        request = json.loads(request.decode('utf-8'))
        if request['type'] == 'request':
            para_name = request['para_name']
            index = request['index']
            if index == None:
                value = network.para[para_name]
            else:
                value = network.para[para_name][index]
            response = {
                'type': 'response',
                'para_name': para_name,
                'index': index,
                'value': value
            }
            response = json.dumps(response).encode('utf-8')
            ws_ps_sock.send(response)

        elif request['type'] == 'push':
            para_name = request['para_name']
            index = request['index']
            if index == None:
                network.para[para_name] = request['value']
            else:
                network.para[para_name][index] = request['value']
            response = {
                'type': 'response',
                'status': 'success'
            }
            response = json.dumps(response).encode('utf-8')
            ws_ps_sock.send(response)

        elif request['type'] == 'command':
            if request['operation'] == 'stop':
                return

def loss_value(network):
    loss_arr = np.array(network.para['expec_outputs']) - network.para['outputs']
    loss_val = 0
    for i in loss_arr:
        loss_val += i ** 2
    loss_val = loss_val / 2
    return loss_val

def error_eval(network, ws_ps_sock, test_x, test_y):
    error_val = 0
    for i in range(len(test_x)):
        network.para['inputs'] = test_x[i]
        network.para['expec_outputs'] = test_y[i]
        test_cmd = {
            'type': 'command',
            'operation': 'run'
        }
        test_cmd = json.dumps(test_cmd, indent=4).encode('utf-8')
        ws_ps_sock.send(test_cmd)
        tackle_ws_request(network, ws_ps_sock)
        
        error_val += loss_value(network)
    return error_val / len(test_x)

def max_index(inputs):
    '''
    return the index of the biggest item in list

    :type inputs: list
    '''
    max_item = -1
    index = 0
    for i in range(len(inputs)):
        if inputs[i] > max_item:
            max_item = inputs[i]
            index = i
    return index

def accuracy(network, ws_ps_sock, test_x, test_y):
    accu = 0
    for i in range(len(test_x)):
        network.para['inputs'] = test_x[i]
        test_cmd = {
            'type': 'command',
            'operation': 'run'
        }
        test_cmd = json.dumps(test_cmd, indent=4).encode('utf-8')
        ws_ps_sock.send(test_cmd)
        tackle_ws_request(network, ws_ps_sock)

        outputs = max_index(network.para['outputs'])
        
        if outputs == max_index(test_y[i]):
            accu += 1
    accu = accu / len(test_y)
    return accu