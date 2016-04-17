__author__ = 'StreetHustling'

import socket
import sys
import thread

def ConnectServer():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = 'localhost'
    port = 6699

    s.bind((host, port))

    s.listen(1)
    connection, addr = s.accept()
    print "Connection from: " + str(addr)
    while True:
        # Wait for a connection
        print >>sys.stderr, 'waiting for a connection'
        connection, client_address = s.accept()
        data = connection.recv(1024)
        try:
            print >>sys.stderr, 'received "%s"' % data
            if data:
                print >>sys.stderr, 'sending data back to the client'
                connection.sendall(data)
            else:
                print sys.stderr, 'no more data from', client_address
                break
        finally:
            # Clean up the connection
            connection.close()

# Create two threads as follows
try:
   thread.start_new_thread( ConnectServer, () )
except:
   print "Error: unable to start thread"

while 1:
   pass
