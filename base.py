import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import msvcrt
import math
import numpy as np
import mqtt

class BaseDNNFlow():


    def __init__(self, #최소 파라미터 : nets, dataset2eval, dataLoader2eval, cb2eval, isTrainMode
                 nets,
                 dataset=None, dataset2eval=None, dataLoader=None, dataLoader2eval=None, #세터로 주입해도 된다.
                 criterion=None, optim=None, # 모델 추론시 criterion, optim은 필요 없음
                 cb2train=None, cb2eval=None,
                 saveDictName=None, loadDictName=None,
                 isTrainMode=True, toBoxing=False,
                 tag=None,
                 msgToPub=None
                 ):

        self.dataset = dataset
        self.dataset2eval = dataset2eval
        self.dataLoader = dataLoader
        self.dataLoader2eval = dataLoader2eval

        self.criterion = criterion
        self.net = nets[1]
        self.netFront = nets[0]
        self.optim = optim
        self.cb2train = cb2train
        self.cb2eval = cb2eval
        self.saveDictName = saveDictName
        self.loadDictName = loadDictName
        self.isTrainMode = isTrainMode
        self.toBoxing = toBoxing
        self.tag = tag

        self.msgToPub = msgToPub

    @property
    def dataset(self):
        return self.__dataset
    @dataset.setter
    def dataset(self, val):
        self.__dataset = val

    @property
    def dataset2eval(self):
        return self.__dataset2eval
    @dataset2eval.setter
    def dataset2eval(self, val):
        self.__dataset2eval = val

    @property
    def dataLoader(self):
        return self.__dataLoader
    @dataLoader.setter
    def dataLoader(self, val):
        self.__dataLoader = val

    @property
    def dataLoader2eval(self):
        return self.__dataLoader2eval
    @dataLoader2eval.setter
    def dataLoader2eval(self, val):
        self.__dataLoader2eval = val

    @property
    def msgToPub(self):
        return self.__msgToPub
    @msgToPub.setter
    def msgToPub(self, val):
        self.__msgToPub = val


    def save(self):
        if self.net == None:
            torch.save(self.netFront.state_dict(), f'{self.saveDictName}.dat')
        else:
            torch.save(self.net.state_dict(), f'{self.saveDictName}.dat')
        return "딥러닝 모델 가중치 저장됨. saved."

    def do(self, epoch=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            pass

        optimizer = self.optim


        def train(net, epoch):
            net.train()
            for batchi, sample in enumerate(self.dataLoader):
                _, x, y = sample

                x = x.to(device)
                y = y.to(device)

                net = net.to(device)

                pred = net(x)


                ''' criterion 직접 변경. 경우에 따라 1, 2, 3가지가 될 수 있음'''
                if isinstance(self.criterion, list):
                    loss = self.criterion[0](pred[:, :4], y[:, :4]) + self.criterion[1](pred[:, 4:8], y[:, 4].long()) # 이런 식으로도 작성 가능하지만 효과가 좋지 않아 사용 정지하였습니다.
                elif self.tag == "hrnet2":
                    loss = self.criterion(pred, torch.sqrt(y))
                else:
                    loss = self.criterion(pred, y)
                ''' 끝 '''

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.cb2train(self, epoch, loss, pred)
                return f'loss={loss}'

        def eval():
            def checkRedPixelCount(im, lbl):
                def check(a, gt=None):
                    r, g, b = a
                    if r > b + g:
                        return 1
                    return 0

                im = im.permute(1, 2, 0).contiguous()
                im = torch.tensor(im, dtype=torch.uint8)
                im = Image.fromarray(im.cpu().numpy())
                if 100 < np.sum(np.array(list(map(lambda x: check(x), im.getdata())))) < 20000:
                    return f'{lbl}'
                return f'*{lbl}'

            # 배경 학습을 별도의 모델에서 할 때 주석을 해제하시고 개발하세요.
            '''
            self.netFront.load_state_dict(torch.load("model2BboxYOLO-Front.dat"))
            self.netFront.eval()
            self.netFront = self.netFront.to(device)
            '''

            self.net.load_state_dict(torch.load(f"{self.loadDictName}.dat"))
            self.net.eval()
            self.net = self.net.to(device)

            with torch.no_grad():
                # 배경 구별시 룰베이스가 아닌 별도의 모델을 이용할 경우 아래 빨강 픽셀을 카운팅 하는 소스코드 3줄을 변경하세요.
                arrXY = self.dataLoader2eval
                for i, im in enumerate(arrXY.dataset.x):
                    arrXY.dataset.y[i] = checkRedPixelCount(im, arrXY.dataset.y[i])

                arrXY.dataset.pred = list(
                    map(lambda x: self.net(torch.unsqueeze(x, 0).to(device)).squeeze(), arrXY.dataset.x))
                self.cb2eval(self, arrXY, self.criterion)

        if self.isTrainMode:
            return train(self.net, epoch)
        else:
            eval()
        pass



