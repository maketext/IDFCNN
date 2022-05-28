import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

import mqtt

import netBasicCNN2NN
# from vit_pytorch import ViT
# from mlp_mixer_pytorch import MLPMixer

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import base
from os import listdir, remove
from os.path import isfile, join
import sys
import Utils.makeBox


basePath = r"img\원-배치-태스크\불량데이터"



def pxToNPArray(arr):
    arr = np.array(arr)
    return arr
def pxToNPArrayUint8(arr):
    arr = np.array(arr, dtype=np.uint8)
    return arr


class FrontDataset(Dataset):

    def __init__(self):
        global basePath
        self.x = []

        self.y = []

        patchFolder = fr"{basePath}/배경/"
        patchFiles = [f for f in listdir(patchFolder) if isfile(join(patchFolder, f))]
        if len(patchFiles) == 0:
            pass
        for (i, filename) in enumerate(patchFiles):
            try:
                im = Image.open(f'{patchFolder}{filename}')
                im = transforms.Resize((224, 224))(im)
                t1 = torch.tensor(pxToNPArray(im), dtype=torch.float32)
                t1 = t1.permute(2, 0, 1).contiguous()
                self.x.append(t1)
                self.y.append([10., 10., 10., 10., 10.])
            except FileNotFoundError as e:
                pass
        print(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return 0, self.x[i], torch.tensor(self.y[i], dtype=torch.float32)

class CustomDataset(Dataset):
    def __init__(self):
        self.x = []
        self.y = []

    def uninit(self):
        self.x = []
        self.y = []

    def isDataExist(self):
        if len(self.y) > 0:
            return True
        return False

    def init(self, type):
        global basePath
        _, cnt, type = Utils.makeBox.makeBox(type)
        for oneY in _:
            self.y.append(oneY)
        folder = type
        if folder == 10:
            folder = 0

        for j in range(1, 200):
            try:
                im = Image.open(f'{basePath}/{folder}/u{j}.jpg')
                im = transforms.Resize((224, 224))(im)
                t1 = torch.tensor(pxToNPArray(im), dtype=torch.float32)
                t1 = t1.permute(2, 0, 1).contiguous()
                self.x.append(t1)
            except FileNotFoundError as e:
                break
        return cnt

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return 0, self.x[i], torch.tensor(self.y[i], dtype=torch.float32)


dataset = CustomDataset()
dataloader = None

frontDataset = FrontDataset()
frontDataloader = DataLoader(
    dataset=frontDataset,
    batch_size=frontDataset.__len__(),
    shuffle=True
)


def cb2train(self, ep, loss, pred):
    print(f'loss={loss}')
    #msvcrt.getch()
    pass

def cb2eval(self, arrXY, criterion__):
    pass

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

# criterion=[nn.MSELoss(), nn.CrossEntropyLoss()]
net = netBasicCNN2NN.MainBL()
dnnflow = base.BaseDNNFlow(
    dataset=dataset, dataset2eval=None,
    dataLoader=dataloader, dataLoader2eval=None,
    criterion=nn.MSELoss(), nets=(net, net),
    optim=optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9),
    cb2train=cb2train, cb2eval=cb2eval,
    saveDictName="model2BboxYOLO", loadDictName="model2BboxYOLO",
    isTrainMode=True
)

cli = mqtt.conn('Train')
count = 0
def on_message(cli, userData, msg_):
    global count, dataloader
    msg = msg_.payload.decode()
    topic = msg_.topic


    if topic == '/start-train-1':
        if not dataset.isDataExist():
            return


    if topic == '/start-train-init':
        mqtt.pub(cli, '/on-start-train-init', qos=0)
        mqtt.pub(cli, '/makebox-0', qos=0)
        mqtt.pub(cli, '/makebox-1', qos=0)
        mqtt.pub(cli, '/makebox-2', qos=0)
        mqtt.pub(cli, '/makebox-3', qos=0)
    elif topic == '/start-train-weight-init':
        netBasicCNN2NN.initialize_weights(dnnflow.net)
        mqtt.pub(cli, '/start-train-init', qos=0)
    elif topic == '/start-train-1':
        count = count + 1
        if count > 1000:
            mqtt.pub(cli, '/on-train', f'EP={count} {dnnflow.do(epoch=msg)}', qos=0)
        else:
            mqtt.pub(cli, '/on-train', f'ep={count} {dnnflow.do(epoch=msg)}', qos=0)


    elif topic == '/start-train-save':
        mqtt.pub(cli, '/on-train', f'{dnnflow.save()}')
        count = 0
    elif topic == '/q':
        print("quit!!")
        sys.exit()
    elif topic == '/makebox-0':
        dataset.uninit()
        print("makebox")
        mqtt.pub(cli, '/on-makebox-0', f'양품 u1.jpg ~ u{dataset.init(10)}.jpg 데이터셋 생성 완료.', qos=0)
    elif topic == '/makebox-1':
        print("makebox")
        mqtt.pub(cli, '/on-makebox-1', f'비품 u1.jpg ~ u{dataset.init(1)}.jpg 데이터셋 생성 완료.', qos=0)
    elif topic == '/makebox-2':
        print("makebox")
        mqtt.pub(cli, '/on-makebox-2', f'비품 u1.jpg ~ u{dataset.init(2)}.jpg 데이터셋 생성 완료.', qos=0)
    elif topic == '/makebox-3':
        print("makebox")
        count = 0
        mqtt.pub(cli, '/on-makebox-3', f'비품 u1.jpg ~ u{dataset.init(3)}.jpg 데이터셋 생성 완료.', qos=0)
        if dataset.isDataExist() and dataloader is None:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=dataset.__len__(),
                shuffle=True
            )
            dnnflow.dataLoader = dataloader
        mqtt.pub(cli, '/on-start-train-init-finish', '생성 완료. 학습을 시작하세요.', qos=0)



if __name__ == '__main__':
    mqtt.pub(cli, '/fileopenTrain', qos=0)
    baseVSCodePath = r'웹서버'

    cli.on_message = on_message
    mqtt.sub(cli, '/q')
    mqtt.sub(cli, '/start-train-init')
    mqtt.sub(cli, '/start-train-weight-init')
    mqtt.sub(cli, '/start-train-1')
    mqtt.sub(cli, '/start-train-save')

    mqtt.sub(cli, '/makebox-0')
    mqtt.sub(cli, '/makebox-1')
    mqtt.sub(cli, '/makebox-2')
    mqtt.sub(cli, '/makebox-3')

    mqtt.run(cli, 'loop_forever') # 블락 지점

    # 배경을 학습하기 위한 객체입니다. mqtt.run(cli, 'loop_forever') 소스코드를 주석 처리 한 후 사용하세요.
    # 배경을 룰 베이스로 판정할 경우 필요가 없습니다.
    # 배경을 메인 딥러닝 모델의 한 라벨로 통합해서 잘 되는 경우 필요가 없습니다.
    frontDnnflow = base.BaseDNNFlow(
        dataset=frontDataset, dataset2eval=None,
        dataLoader=frontDataloader, dataLoader2eval=None,
        criterion=nn.MSELoss(), nets=(net, net),
        optim=optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9),
        cb2train=cb2train, cb2eval=cb2eval,
        saveDictName="model2BboxYOLO-Front", loadDictName="model2BboxYOLO-Front",
        isTrainMode=True
    )
    for i in range(390):
        frontDnnflow.do()
    
    print(frontDnnflow.save())