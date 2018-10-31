# worker server
import socket
import json
import numpy as np

LARGEST_RECV = 2**20
WS_PS_PORT = 10002

# build connection with parameters server
host = socket.gethostname()
ws_ps_sock = socket.socket()
ws_ps_sock.bind((host, WS_PS_PORT))
ws_ps_sock.listen(1)
print('connecting with parameters server...')
ps_sock, addr = ws_ps_sock.accept()
print('connected with parameters server!')


def requset_para(para_name, index=None):
    '''
    request para from ps

    :para para_name: name of para
    :type para_name: str

    :para ps_sock: socket connection between ws and ps, to get paras from ps
    :type ps_sock: socket object

    :para index: the index of requested content in the whole para,
                 return content is para_name[index], if index is None, than return whole para
    :type index: number
    '''
    request = {
        'type': 'request',
        'para_name': para_name,
        'index': index
    }
    request = (json.dumps(request, indent=4)).encode('utf-8')
    ps_sock.send(request)
    response = ps_sock.recv(LARGEST_RECV)
    response = json.loads(response.decode('utf-8'))
    # print(response)
    return response['value']

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
    ps_sock.send(push)
    response = ps_sock.recv(LARGEST_RECV)
    response = json.loads(response.decode('utf-8'))
    if response['status'] == 'success':
        return 0

def sigmiodal(inputs):
    return (np.exp(inputs) / (np.exp(inputs) + 1)).tolist()

# forward propagation
def forward():
    # print('start train with inputs: ', str(requset_para('inputs')))
    hl_inputs = (np.dot(requset_para('inputs'), requset_para('weights', 0))).tolist()
    push_para('hl_inputs', hl_inputs, 0)
    hl_outputs = sigmiodal(requset_para('hl_inputs', 0))
    push_para('hl_outputs', hl_outputs, 0)

    for i in range(1, len(requset_para('sizes')) - 2):
        hl_inputs = (np.dot(requset_para('hl_inputs', i-1), requset_para('weights', i)) - requset_para('biases', i)).tolist()
        push_para('hl_inputs', hl_inputs, i)
        hl_outputs = sigmiodal(requset_para('hl_inputs', i))
        push_para('hl_outputs', hl_outputs, i)
    
    outputs_in = (np.dot(requset_para('hl_outputs', -1), requset_para('outputs_weight')) - requset_para('outputs_bias')).tolist()
    push_para('outputs_in', outputs_in)
    outputs = sigmiodal(requset_para('outputs_in'))
    push_para('outputs', outputs)

if __name__ == '__main__':
    print('worker server is working...')
    while True:
        ps_request = ps_sock.recv(LARGEST_RECV)
        ps_request = json.loads(ps_request.decode('utf-8'))
        # print(ps_request)
        if ps_request['type'] == 'command': 
            if ps_request['operation'] == 'run':
                forward()
                # forward completed, stop and switch to back in ps
                # print('forward completed!')
                stop_request = {
                    'type': 'command',
                    'operation': 'stop'
                }
                stop_request = json.dumps(stop_request, indent=4).encode('utf-8')
                ps_sock.send(stop_request)
            elif ps_request['operation'] == 'stop':
                break
        elif ps_request['type'] == 'info':
            if ps_request['train_info'] == 'done':
                print('worker server work done!')
                exit()