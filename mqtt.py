import paho.mqtt.client as mClient
import threading
import random
brokerAddr = "localhost"
brokerPort = 1883

def conn(id=None) -> mClient:
    if id == None:
        id = random.randint(0, 30000)
    def on_connect(client, userData, flags, rc):
        if rc == 0:
            pass
    def on_disconnect(client, userData, flags, rc=0):
        pass
    clientId = fr'client-python-{id}'
    client = mClient.Client(clientId)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(host=brokerAddr, port=brokerPort)
    return client

def pub(client: mClient, topicTo, msgToPub=None, qos=2):
    res = client.publish(topic=topicTo, payload=msgToPub ,qos=qos)
    stat = res[0]
    if stat == 0:
        print(f"PUB topic={topicTo} 퍼블리시됨.")

def sub(client: mClient, topicFrom, qos=2):
    print(f'sub({topicFrom}) start')
    client.subscribe(topicFrom ,qos=qos)

def run(cli, flag):
    print("무한루프 시작")
    if flag == 'loop_forever':
        cli.loop_forever()
    elif flag == 'loop_start':
        cli.loop_start()

def work(cli, topicFrom, topicTo, onPing): # 블락될 시점 후 미래의 시점.

    sub(client=cli, topicFrom=topicFrom, topicTo=topicTo, cb=onPing)
    print(f'SUB topic={topicFrom}')
    pass

def setNewMQThread(onPing, topicFrom='', topicTo=''):
    cli = conn()
    th = threading.Thread(target=work, args=(cli, topicFrom, topicTo, onPing))
    th.start()
    run(cli, 'loop_forever') # 블락 지점