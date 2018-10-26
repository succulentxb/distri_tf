# worker server
import socket
import json

LARGEST_RECV = 2**16

# waiting for data from master
ws_socket = socket.socket()
host = socket.gethostname()
ws_port = 10001
ws_socket.bind((host, ws_port))
ws_socket.listen(1)
master_socket, addr = ws_socket.accept()
data = master_socket.recv(LARGEST_RECV)

# print(data.decode('utf-8'))
data = data.decode('utf-8')
data = json.loads(data)