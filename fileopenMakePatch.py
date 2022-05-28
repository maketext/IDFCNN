import numpy as np
from PIL import Image
import patching
import mqtt



def pxToNPArray(arr):
    #arr = (1,2,3)
    arr = np.array(arr)
    return tuple(arr)
def pxRestore(arr):
    #arr = (1,2,3)
    arr = np.array(arr)
    arr = list(arr * 25)
    return tuple(arr)


'''MQ 사용자화 코드 부분'''
def onPing(msgFromSub = None):
    global baseVSCodePath

    print("onPing")
    imArr = []

    #imArr.append(Image.open(fr'C:\\cam.jpg').convert("RGB"))
    imArr.append(Image.open(fr'{baseVSCodePath}\res\img\웹캠.jpg').convert("RGB"))

    for i in range(len(imArr)):
        patching.patching(i, imArr[i])

    print("onPing dnnflow.do() end.")
    print("publish.")
    return msgFromSub


cli = mqtt.conn('MakePatch')
def on_message(cli, userData, msg_):
    msg = msg_.payload.decode()
    topic = msg_.topic
    if topic == '/start':
        onPing()
        mqtt.pub(cli, '/makepatch-done', qos=0)
    elif topic == '/q':
        print("quit!!")
        exit()

if __name__ == '__main__':
    baseVSCodePath = r'웹서버'

    mqtt.pub(cli, '/fileopenMakePatch', qos=0)

    cli.on_message = on_message
    mqtt.sub(cli, '/start', qos=0)
    mqtt.sub(cli, '/q', qos=0)
    mqtt.run(cli, 'loop_forever')
