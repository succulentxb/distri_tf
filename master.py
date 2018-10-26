import socket
import json 
import csv

LARGEST_RECV = 2**16

# Waiting for user inputs from client.
master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
master_port = 10000

master_sock.bind((host, master_port))
master_sock.listen(1)

print('waiting for inputs from client...')
client_sock, addr = master_sock.accept()
# print("client addr: %s" % str(addr))
inputs = client_sock.recv(LARGEST_RECV)
print('master recieved inputs:\n%s' % inputs.decode('utf-8'))
client_sock.close()

inputs = json.loads(inputs.decode('utf-8'))

# parse input sizes, '1,2,3' -> [1, 2, 3]
sizes = inputs['sizes'].split(',')
for i in range(len(sizes)):
    sizes[i] = int(sizes[i])
# print(sizes)

train_file = open(inputs['train_file'], newline='')
csvreader = csv.reader(train_file, delimiter=',')
data_list = []
for row in csvreader:
    data_list.append(row)

data_list = data_list[1:]
train_x = []
train_y = []
for data in data_list:
    train_y.append(data[0])
    train_x.append(data[1:])
# print(train_expec_outputs)
# print(train_data_inputs)

# data_ws is data for worker server
data_ws = {
    'sizes': sizes
}

# data_ps is data for parameters server
data_ps = {
    'sizes': sizes,
    'train_data_inputs': train_x,
    'train_': train_y
}

data_ws_json = json.dumps(data_ws, sort_keys=True, indent=4)
data_ps_json = json.dumps(data_ps, sort_keys=True, indent=4)

master_sock = socket.socket()
ws_port = 10001
master_sock.connect((host, ws_port))
master_sock.send(data_ws_json.encode('utf-8'))