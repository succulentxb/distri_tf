import socket
import json
import network_ps as nt 
import numpy as np

LARGEST_RECV = 2**16
PS_MASTER_PORT = 10001
WS_PS_PROT = 10002

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

if __name__ == '__main__':
    host = socket.gethostname()

    # build connection with worker server
    ws_ps_sock = socket.socket()
    ws_ps_sock.connect((host, WS_PS_PROT))

    # build connection with master
    ps_master_sock = socket.socket()
    ps_master_sock.bind((host, PS_MASTER_PORT))
    ps_master_sock.listen(1)
    master_sock, master_addr = ps_master_sock.accept()

    print('waiting for data from master...')
    train_data = master_sock.recv(LARGEST_RECV)
    # print(train_data.decode('utf-8'))
    train_data = json.loads(train_data.decode('utf-8'))
    train_x = train_data['train_x']
    train_y = train_data['train_y']

    # initialize the bp network
    network = nt.Network(train_data['sizes'])

    # start train
    train_upper = 100 # total train time
    for train_time in range(train_upper):
        for i in range(len(train_x)):
            network.para['inputs'] = train_x[i]
            network.para['expec_outputs'] = train_y[i]
            # print('train with inputs: ', str(train_x[i]))
            train_cmd = {
                'type': 'command',
                'operation': 'train'
            }
            train_cmd = json.dumps(train_cmd, indent=4).encode('utf-8')
            ws_ps_sock.send(train_cmd)
            tackle_ws_request(network, ws_ps_sock)
            # print('back start!')
            back(network)
            # print(network.para['outputs'])
        print('train time: ', str(train_time + 1), ', loss value: ', str(loss_value(network)))
        #print(network.para['outputs'])
    
    while True:
        pass