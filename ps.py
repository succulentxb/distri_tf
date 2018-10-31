import socket
import json
import network_ps as nt 
import numpy as np

LARGEST_RECV = 2**20
PS_MASTER_PORT = 10001
WS_PS_PROT = 10002

if __name__ == '__main__':
    host = socket.gethostname()

    # build connection with worker server
    print('connecting with worker server...')
    ws_ps_sock = socket.socket()
    ws_ps_sock.connect((host, WS_PS_PROT))
    print('worker server connected!')

    # build connection with master
    ps_master_sock = socket.socket()
    ps_master_sock.bind((host, PS_MASTER_PORT))
    ps_master_sock.listen(1)
    print('waiting for connection from master...')
    master_sock, master_addr = ps_master_sock.accept()
    print('master connected!')

    print('waiting for data from master...')
    master_data = master_sock.recv(LARGEST_RECV)
    print('recieved data from master, start parsing data...')
    master_data = json.loads(master_data.decode('utf-8'))
    train_x = master_data['train_x']
    train_y = master_data['train_y']

    test_x = master_data['test_x'][:100]
    test_y = master_data['test_y'][:100]

    # initialize the bp network
    network = nt.Network(master_data['sizes'], 0.3) 

    print('start training network...')
    # start train
    train_upper = 100 # total train time
    for train_time in range(train_upper):
        for i in range(len(train_x)):
            network.para['inputs'] = train_x[i]
            network.para['expec_outputs'] = train_y[i]
            # print('train with inputs: ', str(train_x[i]))
            train_cmd = {
                'type': 'command',
                'operation': 'run'
            }
            train_cmd = json.dumps(train_cmd, indent=4).encode('utf-8')
            ws_ps_sock.send(train_cmd)
            nt.tackle_ws_request(network, ws_ps_sock)
            # print('back start!')
            nt.back(network)
            # print(network.para['outputs'])
        accu = nt.accuracy(network, ws_ps_sock, test_x, test_y)
        train_info_str = 'train time: {:d}, accuracy: {:.2f}%'.format(train_time, accu*100)
        train_info = {
            'type': 'info',
            'train_info': train_info_str
        }
        train_info = json.dumps(train_info).encode('utf-8')
        master_sock.send(train_info)
        print(train_info_str)
        #print(network.para['outputs'])
    
