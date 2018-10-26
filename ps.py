import socket
import json
import network_ps as nt 

LARGEST_RECV = 2**16

# waiting for data from master
ps_sock = socket.socket()
host = socket.gethostname()
ps_port = 10002
ps_sock.bind((host, ps_port))
ps_sock.listen(1)
print('waiting for data from master...')
master_sock, master_addr = ps_sock.accept()

data = master_sock.recv(LARGEST_RECV)
data = data.decode('utf-8')
data = json.loads(data)

# initialize the bp network
network = nt.Network(data['sizes'])

# waiting for request from worker server
ps_ws_sock = socket.socket()
ps_ws_port = 10003
ps_ws_sock.bind((host, ps_ws_sock))
ws_sock, ws_addr = ps_ws_sock.accept()
while True:
    # recieve worker server request
    ws_request = ws_sock.recv(LARGEST_RECV)
    ws_request = json.loads(ws_request.decode('utf-8'))
    # if ws_request['request'] == '':
    