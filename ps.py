import socket
import json
import network_ps as nt 
import numpy as np

LARGEST_RECV = 2**16

# waiting for data from master
ps_sock = socket.socket()
host = socket.gethostname()
ps_port = 10002
ps_sock.bind((host, ps_port))
ps_sock.listen(1)
print('waiting for data from master...')
master_sock, master_addr = ps_sock.accept()

data = master_sock.recv(LARGEST_RECV)
data = data.decode('utf-8')
data = json.loads(data)

# initialize the bp network
network = nt.Network(data['sizes'])

# waiting for request from worker server
ps_ws_sock = socket.socket()
ps_ws_port = 10003
ps_ws_sock.bind((host, ps_ws_sock))
ws_sock, ws_addr = ps_ws_sock.accept()
while True:
    # recieve worker server request
    ws_request = ws_sock.recv(LARGEST_RECV)
    ws_request = json.loads(ws_request.decode('utf-8'))
    # if ws_request['request'] == '':

def sigmiodal_deri(inputs):
    tmp = np.exp(inputs) / (np.exp(inputs) + 1)
    return (tmp * (1-tmp)).tolist()

def back(network):
    network.para['outputs_deris'] = (np.array(network.para['outputs']) - network.para['expec_outputs']).tolist()
    
    network.para['hl_out_deris'][-1] = np.dot(network.para['outputs_weight'], network.para['outputs_deris'])
    network.para['thf_deris'][-1] = sigmiodal_deri(network.para['hl_inputs'][-1])
    network.para['thf_out_products'][-1] = (np.array(network.para['thf_deris'][-1]) * network.para['hl_out_deris'][-1]).tolist()

    for i in range(len(network.para['hl_outputs'])-2, -1, -1):
        network.para['hl_out_deris'][i] = (np.dot(network.para['weights'][i+1], network.para['thf_out_products'][i+1])).tolist()
        network.para['thf_deris'][i] = sigmiodal_deri(network.para['hl_inputs'][i])
        network.para['thf_out_products'][i] = (np.array(network.para['thf_deris'][i]) * network.para['hl_out_deris'][i]).tolist()

        # compute weight derivatives of output layer
        row = len(network.para['hl_outputs'][-1])
        col = len(network.para['outputs_deris'])
        network.para['outputs_weight_deris'] = (np.dot(np.array(network.para['hl_outputs'][-1]).reshape(row, 1), 
                                            np.array(network.para['outputs_deris']).reshape(1, col))).tolist()
        
        # compute weight derivatives of input layer
        row = len(network.para['inputs'])
        col = len(network.para['thf_out_products'][0])
        network.para['weight_deris'][0] = (np.dot(np.array(network.para['inputs']).reshape(row, 1), 
                                            np.array(network.para['thf_out_deris'][0]).reshape(1, col))).tolist()
        
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
        network.para['outputs_bias_deris'] = (np.array(network.para['outputs_deris']) * (-1)).tolist()

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