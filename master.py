import socket
import json 
import csv

LARGEST_RECV = 2**16
MASTER_CLIENT_PROT = 10000
PS_MASTER_PORT = 10001

if __name__ == '__main__':
    host = socket.gethostname()
    # Build a server for client.
    master_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    master_client_sock.bind((host, MASTER_CLIENT_PROT))
    master_client_sock.listen(1)
    # Build a server for parameters server.
    master_ps_sock = socket.socket()

    # connection with ps
    print('connecting with parameters server...')
    master_ps_sock.connect((host, PS_MASTER_PORT))
    print('connected with parameters server!')

    # connection with client
    print('connecting with client...')
    client_sock, client_addr = master_client_sock.accept()
    print('connected with client!')
    
    # waiting input data from client
    user_inputs = client_sock.recv(LARGEST_RECV)
    print('master recieved inputs:\n%s' % user_inputs.decode('utf-8'))    

    user_inputs = json.loads(user_inputs.decode('utf-8'))

    # parse input sizes, '1,2,3' -> [1, 2, 3]
    sizes = user_inputs['sizes'].split(',')
    for i in range(len(sizes)):
        sizes[i] = int(sizes[i])
    # print(sizes)

    # read train data from file
    train_file = open(user_inputs['train_file'], newline='')
    csvreader = csv.reader(train_file, delimiter=',')
    data_list = []
    for row in csvreader:
        data_list.append(row)

    data_list = data_list[1:] # remove first header row
    train_x = []
    train_y = []
    for data in data_list:
        train_y.append(data[0])
        train_x.append(data[1:])

    data_ps = {
        'operation': 'train',
        'sizes': sizes,
        'train_x': train_x,
        'train_y': train_y
    }
    data_ps = json.dumps(data_ps, indent=4).encode('utf-8')

    master_ps_sock.send(data_ps)

    for i in range(len(train_x)):
        # recieve train information from parameters server
        train_info = master_ps_sock.recv(LARGEST_RECV)
        train_info_decode = json.loads(train_info.decode('utf-8'))
        # if training normally, print and send info to client
        if train_info_decode['type'] == 'info':
            print('train time: ', train_info_decode['train_time'])
            master_client_sock.send(train_info)