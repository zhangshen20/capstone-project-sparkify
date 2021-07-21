import socket
import sys
import requests
import requests_oauthlib
import json
import time
from random import seed
from random import randint

def send_tweets_to_spark(http_resp, tcp_connection):
    '''Send action events via sockets
    '''

    try:
        # tweet_data = bytes(http_resp + '\n', 'utf-8')
        tweet_data = bytes(http_resp, 'utf-8')
        tcp_connection.send(tweet_data)
    except:
        e = sys.exc_info()[0]
        print("Error: %s" % e)

def load_event():
    '''Load Sparkify User action events
    '''

    dataList = []
    # with open('../../mini_sparkify_event_data.json') as f:
    with open('../../sparkify_event_data_2.json') as f:    

        for jsonObj in f:
            data = json.loads(jsonObj)
            dataList.append(data)

    return dataList

# def load_event():
#     '''Load Sparkify User action events
#     '''

#     with open('../../sparkify_event_data_2.json') as f:
#         contents = f.readlines()

#     return contents

resp = load_event()

TCP_IP = "localhost"
TCP_PORT = 3001

conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
print("Waiting for TCP connection...")
conn, addr = s.accept()
print("Connected... Starting sending events.")

seed(1)
n_samples = len(resp)

value = 0
offset = 1000

n_times = int(2000000/offset)

for _ in range(n_times):

    msg = ''

    for _ in range(offset):

        msg = msg + json.dumps(resp[value]) + '\n'
        # msg = msg + resp[value] + '\n'
        value = value + 1

    send_tweets_to_spark(msg,conn)
    time.sleep(0.5)