import socket
import json 
import csv

LARGEST_RECV = 2**20
MASTER_CLIENT_PROT = 10000
PS_MASTER_PORT = 10001

def csv_to_data(file_name):
    '''
    read data from csv file, and return 2 lists, the first one is x, the second one is y

    :para file_name: name of csv file
    :type file_name: str
    '''
    with open(file_name, newline='') as file:
        csvreader = csv.reader(file, delimiter=',')
        data_list = []
        for row in csvreader:
            data_list.append(row)
        
        # remove first header row
        data_list = data_list[1:]
        # transfer the number str to float
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j] = float(data_list[i][j])
        
        # preprocess the data, fit data to range 0-1
        for i in range(len(data_list)):
            data_list[i][1] /= 800
            data_list[i][2] /= 4
            data_list[i][3] /= 4
        
        x = []
        y_raw = []
        for data in data_list:
            y_raw.append(data[0])
            x.append(data[1:])
        
        y = []
        for data in y_raw:
            if data == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])

        return x, y
        

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
    print('waiting user input data from client...')
    user_inputs = client_sock.recv(LARGEST_RECV)
    print('recieved data from client, start parsing data...')    

    user_inputs = json.loads(user_inputs.decode('utf-8'))
    sizes = user_inputs['sizes']

    # read train data from file
    train_x, train_y = csv_to_data(user_inputs['train_file'])
    # read test data
    test_x, test_y = csv_to_data(user_inputs['test_file'])
    '''
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)
    '''

    data_ps = {
        'operation': 'train',
        'sizes': sizes,
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y
    }
    data_ps = json.dumps(data_ps, indent=4).encode('utf-8')

    master_ps_sock.send(data_ps)
    print('sent train data to parameters server.')
    for i in range(len(train_x)):
        # recieve train information from parameters server
        train_info = master_ps_sock.recv(LARGEST_RECV)
        train_info_decode = json.loads(train_info.decode('utf-8'))
        # if training normally, print and send info to client
        if train_info_decode['type'] == 'info':
            print(train_info_decode['train_info'])
            client_sock.send(train_info)
    train_info = {
        'type': 'info',
        'train_info': 'done'
    }
    train_info = json.dumps(train_info).encode('utf-8')
    client_sock.send(train_info)
    print('train done!')
    