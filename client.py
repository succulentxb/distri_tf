import socket
import json
import re

MASTER_CLIENT_PORT = 10000
LARGEST_RECV = 2**20

if __name__ == '__main__':
    
    # Get input data from user. If input with wrong fomat, exit.
    '''
    sizes = input('please enter the network structure like 3,10,1:\n')
    if not re.match(r'([0-9]+,){2,}[0-9]+', sizes):
        print('please enter a correct struct format like 1,2,3 with no space.')
        exit()
    '''
    '''
    train_file = input('please enter csv file name of train set:\n')
    if not re.match(r'\w+\.csv', train_file):
        print('please enter a correct csv file name.')
        exit()

    test_file = input('please enter csv file name of test set:\n')
    if not re.match(r'\w+\.csv', test_file):
        print('please enter a correct csv file name.')
        exit()

    inputs = {
        'sizes': [3, 10, 2],
        'train_file': train_file,
        'test_file': test_file
    }
    '''
    inputs = {
        'sizes': [3, 10, 2],
        'train_file': 'train.csv',
        'test_file': 'test.csv',
        'train_time': 100
    }
    
    inputs_json = json.dumps(inputs, indent=4)
    # print('accept input: %s' % inputs_json)

    # Send user input to master.
    print('sending your input to master...')
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    client_sock.connect((host, MASTER_CLIENT_PORT))
    client_sock.send(inputs_json.encode('utf-8'))
    print('input sent, please waiting for result patiently.')

    # recieve 
    while True:
        train_info = client_sock.recv(LARGEST_RECV)
        train_info = json.loads(train_info.decode('utf-8'))
        print(train_info['train_info'])
        if train_info['train_info'] == 'done':
            print('train done!')
            break
    
    while True: 
        gre = input('enter gre grades (200-800): ')
        gpa = input('enter gpa (0.0-4.0): ')
        rank = input('enter rank (1-4): ')

        inputs = [float(gre), float(gpa), float(rank)]
        compute_data = {
            'type': 'command',
            'operation': 'compute',
            'inputs': inputs
        }
        compute_data = json.dumps(compute_data, indent=4).encode('utf-8')
        client_sock.send(compute_data)
        print('inputs sent, please wait for result...')
        result = client_sock.recv(LARGEST_RECV)
        result = json.loads(result.decode('utf-8'))
        print('result: ', result['result'])