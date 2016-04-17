__author__ = 'StreetHustling'

import socket
import base64
import time

def Main():
    host = '127.0.0.1'
    port = 6000

    s = socket.socket()
    s.bind((host, port))

    s.listen(1)
    c, addr = s.accept()
    print "Connection from: " + str(addr)
    while True:
        data = c.recv(10000)
        if not data:
            break
        print "from connected user"


        file_name = str(time.strftime("%x")) + ""+ str(time.strftime("%X"))+".jpg"


        FILE = open(file_name,"wb")
        FILE.write(data)
        FILE.close()
    c.close()


if __name__ == "__main__":
    Main()
