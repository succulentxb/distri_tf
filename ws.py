# worker server
import socket
import json
import numpy as np

LARGEST_RECV = 2**16
'''
# waiting for data from master
ws_socket = socket.socket()
host = socket.gethostname()
ws_port = 10001
ws_socket.bind((host, ws_port))
ws_socket.listen(1)
master_socket, addr = ws_socket.accept()
data = master_socket.recv(LARGEST_RECV)

# print(data.decode('utf-8'))
data = data.decode('utf-8')
data = json.loads(data)
'''
ws_ps_sock = socket.socket()
sizes = [1, 1, 1]

def requset_para(para_name, index=None):
    '''
    request para from ps

    :para para_name: name of para
    :type para_name: str

    :para ws_ps_sock: socket connection between ws and ps, to get paras from ps
    :type ws_ps_sock: socket object

    :para index: the index of requested content in the whole para,
                 return content is para_name[index], if index is None, than return whole para
    :type index: number
    '''
    request = {
        'type': 'request',
        'para_name': para_name,
        'index': index
    }
    request = (json.dumps(request, sort_keys=True, indent=4)).encode('utf-8')
    ws_ps_sock.send(request)
    response = ws_ps_sock.recv(LARGEST_RECV)
    response = json.loads(response.decode('utf-8'))
    return response['content']

def push_para(para_name, value, index=None):
    '''
    push paras to parameters server

    :para value: value of the para
    :type value: almostly list
    '''
    push = {
        'type': 'push',
        'para_name': para_name,
        'index': index,
        'value': value
    }
    push = (json.dumps(push, sort_keys=True, indent=4)).encode('utf-8')
    ws_ps_sock.send(push)

def sigmiodal(inputs):
    return (np.exp(inputs) / (np.exp(inputs) + 1)).tolist()

def sigmiodal_deri(inputs):
    tmp = np.array(sigmiodal(inputs))
    return (tmp * (1-tmp)).tolist()

# forward propagation
def forward():
    '''
    :para nn_ws: network for worker server
    '''
    hl_inputs = (np.dot(requset_para(ws_ps_sock, 'inputs'), requset_para('weights', 0))).tolist()
    push_para('hl_inputs', hl_inputs, 0)
    hl_outputs = sigmiodal(requset_para('hl_inputs', 0))
    push_para('hl_outputs', hl_outputs, 0)

    for i in range(1, len(sizes) - 2):
        hl_inputs = (np.dot(requset_para('hl_inputs', i-1), requset_para('weights', i)) - requset_para('biases', i)).tolist()
        push_para('hl_inputs', hl_inputs, i)
        hl_outputs = sigmiodal(requset_para('hl_inputs', i))
        push_para('hl_outputs', hl_outputs, i)
    
    outputs = (np.dot(requset_para('hl_output', -1), requset_para('outputs_weight')) - requset_para('outputs_bias')).tolist()
    push_para('outputs', outputs)