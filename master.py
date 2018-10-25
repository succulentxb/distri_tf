import socket
import json 
import csv

biggest_input = 2**16

# Waiting for user inputs from client.
master_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
master_port = 10000

master_sock.bind((host, master_port))
master_sock.listen(1)

print('waiting for inputs from client...')
client_sock, addr = master_sock.accept()
# print("client addr: %s" % str(addr))
inputs = client_sock.recv(biggest_input)
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
train_data_inputs = []
train_expec_outputs = []
for data in data_list:
    train_expec_outputs.append(data[0])
    train_data_inputs.append(data[1:])
# print(train_expec_outputs)
# print(train_data_inputs)

# data_ws is data for worker server
data_ws = {
    'sizes': sizes
}