import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

import netBasicCNN2NN
#from vit_pytorch import ViT
#from mlp_mixer_pytorch import MLPMixer

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import msvcrt
import math
import mqtt
import base
from Utils.cfg import exePath
from os import listdir, remove
from os.path import isfile, join



basePath = fr"{exePath}img\원-패치-태스크\불량데이터"

cli = mqtt.conn('BboxYOLO')


def pxToNPArray(arr):
    #arr = (1,2,3)
    arr = np.array(arr)
    return arr
def pxToNPArrayUint8(arr):
    #arr = (1,2,3)
    arr = np.array(arr, dtype=np.uint8)
    return arr

class CustomDataset2eval(Dataset):
    def __init__(self):
        self.x = []
        self.y = []
        self.pred = []

        patchFolder = f"{exePath}img/원-배치-태스크/"
        patchFiles = [f for f in listdir(patchFolder) if isfile(join(patchFolder, f))]
        if len(patchFiles) == 0:
            im = Image.new("RGB", (224, 224), (0, 0, 0))
            t1 = torch.tensor(pxToNPArray(im), dtype=torch.float32)
            t1 = t1.permute(2, 0, 1).contiguous()
            self.x.append(t1)
            self.y.append("dummy")

        for (i, filename) in enumerate(patchFiles):
            try:
                im = Image.open(f'{patchFolder}{filename}')
                im = transforms.Resize((224, 224))(im)
                t1 = torch.tensor(pxToNPArray(im), dtype=torch.float32)
                t1 = t1.permute(2, 0, 1).contiguous()
                self.x.append(t1)
                self.y.append(f"{filename}")
            except FileNotFoundError as e:
                pass

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.pred[i], self.x[i], self.y[i]

#업데이트가 안됨에 주의한다. 일회용이다. 생성자에서 초기화되기 때문이다.
dataset2eval = CustomDataset2eval()
dataloader2eval = DataLoader(
    dataset=dataset2eval,
    batch_size=dataset2eval.__len__(),
    shuffle=False
)

def cb2train(self, ep, loss, pred):
    pass
def cb2eval(self, arrXY, criterion__):
    def ceil(n):
        return math.ceil(n)

    def getCenterPoint(mx, my, sze, len):
        return (10, my // 2 - sze // 2)
    def isEqualLbls(tl, i): # 라벨 스레시홀드 값
        if -30 <= tl.float() <= 3.5 and i == 1:
            return True
        if 3.5 < tl.float() <= 20 and i == 10:
            return True
        return False
    def removeFolder(path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for i, file in enumerate(files):
            try:
                remove(f'{path}{file}')
            except FileNotFoundError as e:
                pass


    with torch.no_grad():

        que = {}
        badQue = {}
        removeFolder(f'{exePath}img/원-배치-태스크/유효패치-판정/')

        for batchi, sample in enumerate(arrXY):
            pred, x, y = sample
            x = x.permute(0, 2, 3, 1).contiguous()
            isBad = False

            for j in range(len(x)):
                im = transforms.ToPILImage()(pxToNPArrayUint8(x[j]))
                im = transforms.Resize((224, 224))(im)

                mx = im.size[0]
                my = im.size[1]
                X = mx / 11
                Y = my / 11
                [x1, y1, x2, y2] = [pred[j][0], pred[j][1], pred[j][2], pred[j][3]]
                [x1, y1, x2, y2] = [ceil(x1 * X), ceil(y1 * Y), ceil(x2 * X), ceil(y2 * Y)]
                draw = ImageDraw.Draw(im) #16 28
                area = (y2 - y1) * (x2 - x1)
                if y2 - y1 < 0 or x2 - x1 < 0:
                    area = -1
                if area < 5000:
                    area = -1
                pad = 224

                if y1 > 224+pad or x1 > 224+pad or x2 > 224+pad or y2 > 224+pad:
                    area = -1
                if x1 < 0-pad or y1 < 0-pad or x2 < 0-pad or y2 < 0-pad:
                    area = -1

                tl = pred[j][4]
                lbls = "라벨없음"
                if area == -1:
                    lbls = "라벨없음"
                elif isEqualLbls(tl, 1):
                    lbls = "크랙"
                    isBad = True
                    badQue[y[j]] = True
                elif isEqualLbls(tl, 10):
                    lbls = "양품"

                if y[j].startswith("*"):
                    lbls = '배경'

                font = ImageFont.truetype(r'C:\Windows\Fonts\malgun.ttf', 20)
                draw.text(getCenterPoint(mx, my, 12, len("넓이/200")),
                          f"{y[j].replace('.jpg', '')}\n{lbls}\n{round(tl.float().item(), 2)}", font=font, fill=(0, 0, 255), stroke_width=0)
                draw.rectangle(((x1, y1),(x2, y2)), outline=(255,0,101), width=3)

                im.save(f'img/원-배치-태스크/유효패치-판정/u{j}.jpg')
                que[y[j]] = im.copy()
                torch.set_printoptions(precision=2, sci_mode=False)
            print("개별 패치 저장 완료.")
            imw = 224
            imh = 224
            dst = Image.new('RGBA', (imw * 6+60, imh * 6+60), (255, 255, 255))
            #draw = ImageDraw.Draw(dst)

            def paste(im, xi, yi):
                if xi > 2 and yi > 2:
                    dst.paste(im, (imw * xi+60, imh * yi+60))
                elif xi > 2:
                    dst.paste(im, (imw * xi+60, imh * yi+30))
                elif yi > 2:
                    dst.paste(im, (imw * xi+30, imh * yi+60))
                else:
                    dst.paste(im, (imw * xi+30, imh * yi+30))
                #draw.rectangle(((imw * xi, imh * yi), (imw * (xi + 1), imh * (yi + 1))), outline=(255, 0, 101), width=2)

            xi = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
            yi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
            zi = [3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
            index = 0

            keys = ['[idx0][0rad]0.0.jpg',
            '[idx0][0rad]0.1.jpg',
            '[idx0][0rad]0.2.jpg',
            '[idx0][0rad]1.0.jpg',
            '[idx0][0rad]1.1.jpg',
            '[idx0][0rad]1.2.jpg',
            '[idx0][0rad]2.0.jpg',
            '[idx0][0rad]2.1.jpg',
            '[idx0][0rad]2.2.jpg',

            '[idx0][22rad]0.0.jpg',
            '[idx0][22rad]0.1.jpg',
            '[idx0][22rad]0.2.jpg',
            '[idx0][22rad]1.0.jpg',
            '[idx0][22rad]1.1.jpg',
            '[idx0][22rad]1.2.jpg',
            '[idx0][22rad]2.0.jpg',
            '[idx0][22rad]2.1.jpg',
            '[idx0][22rad]2.2.jpg']
            for k in keys:
                try:
                    paste(que[k], xi[index], yi[index])
                except:
                    paste(que[f'*{k}'], xi[index], yi[index])
                    pass
                index = index + 1
            index = 0

            def convert_rgb_to_rgba(im):
                w, h = im.size
                ima = Image.new(mode="RGBA", size=(w, h))
                res = list(ima.getdata())

                im = list(im.getdata())

                for i in range(w * h):
                    res[i] = (im[i][0], im[i][1], im[i][2], 128)
                ima.putdata(res)
                return ima

            for k in keys:
                try:
                    if badQue[k] == True:
                        paste(que[k], xi[index], zi[index])
                except:
                    try:
                        paste(convert_rgb_to_rgba(que[k]), xi[index], zi[index])
                    except:
                        print("exception from paste.")
                        pass
                    pass
                index = index + 1

            print("이미지 저장 직전")
            '''
            try:
                remove(fr'{baseVSCodePath}\res\img\최종-bad.png')
                remove(fr'{baseVSCodePath}\res\img\최종-good.png')
            except FileNotFoundError as e:
                pass
            '''
            if isBad == True:
                #dst.save(r'C:\Users\Multifunctional_0\Documents\광교\res\img\최종-bad.jpg')
                mqtt.pub(cli, '/bad', '.')
                dst.save(fr'{baseVSCodePath}\res\img\최종-bad.png')
            else:
                #dst.save(r'C:\Users\Multifunctional_0\Documents\광교\res\img\최종-good.jpg')
                mqtt.pub(cli, '/good', '.')
                dst.save(fr'{baseVSCodePath}\res\img\최종-good.png')
            print("이미지 저장 후")

#criterion=[nn.MSELoss(), nn.CrossEntropyLoss()]

'''
net = ViT(
    image_size=224,
    patch_size=32,
    num_classes=5,
    dim=512,
    depth=6,
    heads=16,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)
'''

'''
net = MLPMixer(
    image_size = 224,
    channels = 3,
    patch_size = 16,
    dim = 64,
    depth = 3,
    num_classes = 5
)
'''

net = netBasicCNN2NN.MainBL()
dnnflow = base.BaseDNNFlow(
    dataset=None, dataset2eval=dataset2eval,
    dataLoader=None, dataLoader2eval=dataloader2eval,
    criterion=nn.MSELoss(), nets=(net, net),
    optim=optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9),
    cb2train=cb2train, cb2eval=cb2eval,
    saveDictName="model2BboxYOLO", loadDictName="model2BboxYOLO",
    isTrainMode=False
)

''' MQ 사용자화 코드 부분'''
def onPing(msgFromSub=None):
    print("dnnflow.do()")
    dnnflow.dataset2eval = CustomDataset2eval()
    dnnflow.dataLoader2eval = DataLoader(
        dataset=dnnflow.dataset2eval, # 이 부분에 주의
        batch_size=dataset2eval.__len__(),
        shuffle=True
    )

    dnnflow.do()
    pass

def on_message(cli, userData, msg_):
    msg = msg_.payload.decode()
    topic = msg_.topic
    if topic == '/makepatch-done':
        onPing()
    elif topic == '/q':
        print("quit!!")
        exit()

if __name__ == '__main__':
    baseVSCodePath = r'웹서버'


    mqtt.pub(cli, '/fileopenBboxYOLO', qos=0)

    cli.on_message = on_message
    mqtt.sub(cli, '/makepatch-done', qos=0)
    mqtt.sub(cli, '/q', qos=0)
    mqtt.run(cli, 'loop_forever')
