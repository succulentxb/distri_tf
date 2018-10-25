import socket
import json
import re

# Get input data from user. If input with wrong fomat, exit.
sizes = input('please enter the network structure like 3,10,1:\n')
if not re.match('([0-9]+,){2,}[0-9]+', sizes):
    print('please enter a correct struct format like 1,2,3 with no space.')
    exit()

train_file = input('please enter csv file name of train set:\n')
if not re.match(r'\w+\.csv', train_file):
    print('please enter a correct csv file name.')
    exit()

test_file = input('please enter csv file name of test set:\n')
if not re.match(r'\w+\.csv', test_file):
    print('please enter a correct csv file name.')
    exit()

inputs = {
    'sizes': sizes,
    'train_file': train_file,
    'test_file': test_file
}

inputs_json = json.dumps(inputs, sort_keys=True, indent=4)
print('accept input: %s' % inputs_json)

# Send user input to master.
print('sending your input to master...')
client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()
master_port = 10000
client_sock.connect((host, master_port))
client_sock.send(inputs_json.encode('utf-8'))
client_sock.close()
print('input sent, please waiting for result patiently.')