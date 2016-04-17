__author__ = 'StreetHustling'


import socket
import sys
import os
import base64

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

host= '127.0.0.1'
port=int(2000)
s.bind((host,port))
s.listen(1)

conn,addr =s.accept()

print (conn,addr)

data = conn.recv(1000000)
# data=data.decode("utf-8")
base = base64.b64decode(data)

FILE = open("test10.jpg","wb")
FILE.write(base)
FILE.close()