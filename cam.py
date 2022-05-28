import cv2
import paho.mqtt.client as mClient
import time
import threading
import random
import os
from enum import Enum
from sys import exit
from pathlib import Path

baseVSCodePath = "웹서버"
class KeyCode(Enum):
    Enter = 13

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

brokerAddr = "127.0.0.1"
brokerPort = 1883

def conn() -> mClient:
    def on_connect(client, userData, flags, rc):
        if rc == 0:
            print("연결됨")
            pass
    def on_disconnect(client, userData, flags, rc=0):
        pass
    clientId = f'cli-python-cam'
    client = mClient.Client(clientId)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    client.connect(host=brokerAddr, port=brokerPort)
    return client

filename = fr'{Path.home()}/Pictures/cam.jpg'
filename = fr'{baseVSCodePath}\res\img\웹캠.jpg'

cap = None
count = 0

def setCam(cap, w, h):
    #cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 1)
    #cap.set(cv2.CAP_PROP_FOCUS, 8)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    return cap

#changRes(1280, 720)
#changRes(3264, 2448)

def cam(isSave = True):
    global cap
    if cap is None:
        print("캡쳐 객체가 없습니다.")
        return
    if not cap.isOpened():
        print("카메라가 종료되었습니다.")
        return
    print(f'width :{cap.get(3)}, height : {cap.get(4)} focus : {cap.get(cv2.CAP_PROP_FOCUS)} ({count})')

    ret, frame = cap.read()    # Read 결과와 frame

    if ret:
        #frame = frame[:, 280:1000]  ## 정사각 aspect ratio 적용
        #frame = frame[:,408:2856]
        #frame = frame[:, 840:3000]
        if isSave:
            imwrite(filename, frame)
            pub('/img-save-ready')

    #cv2.destroyAllWindows()

def pub(topic):
    res = client.publish(topic=topic, qos=0)
    stat = res[0]
    if stat == 0:
        pass
        print(f"PUB topic={topic} 퍼블리시됨.")

def on_message(_, userData, msg):
    global cap, count
    if msg.topic == '/img-start':
        print("카메라 시작")
        if cap is None:
            cap = cv2.VideoCapture()
            cap.open(cv2.CAP_DSHOW)
            setCam(cap, 1280, 720)
            # setCam(cap, 3840, 2160) ## 로지텍 BRIO 4K 웹캠의 에스펙트 레이시오입니다.
            # setCam(cap, 3264, 2448) ## QSENN QC4K 웹캠 의 에스펙트 레이시오입니다.
            # setCam(cap, 1920, 1080)
            # setCam(cap, 1280, 720)

        pub('/img-start-ready')
    elif msg.topic == '/img-recv':
        cam()
    elif msg.topic == '/img-stop':
        count = count + 1
        print("카메라 종료")
        if cap is not None:
            cap.set(cv2.CAP_PROP_FPS, 0)
            if count % 100 == 0:
                print(f"카메라 객체 반환됨. 100주기 반환 카운트({count})")
                cap.release() # 카메라 객체 릴리즈 시 QSENN QC4K 웹캠에서 딸깍 거리는 기계식 스위치 소리가 납니다. 스위치는 내구도가 약하므로 최대한 릴리즈를 덜 하는 방향으로 소스코드를 수정하였습니다.
                cap = None

    elif msg.topic == '/cam-release':
        if cap is not None:
            cap.release()
        cap = None
        print("카메라 객체 반환됨.")
    else:
        pass

def sub(topic):
    print(f'sub: MQ {topic}')
    client.subscribe(topic, qos=0)


def run(client, flag):
    sub('/img-recv')
    sub('/img-start')
    sub('/img-stop')
    sub('/cam-release')
    print("무한루프 시작")

    if flag == 'loop_forever':
        client.loop_forever()
    elif flag == 'loop_start':
        client.loop_start()

client = conn()
client.on_message = on_message
run(client, 'loop_forever')