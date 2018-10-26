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
master_sock, addr = ps_sock.accept()

data = master_sock.recv(LARGEST_RECV)
data = data.decode('utf-8')
data = json.loads(data)

network = nt.Network(data['sizes'])

