from PIL import Image
from torchvision import transforms
import Utils.knn as knn
import Utils.cfg as cfg

def setThresHold(im):
    threshold = (230, 50, 50) ## 학습 데이터 m 접두어 jpg 이미지에 칠해진 빨강색 박스 테두리를 검출하기 위한 스레시홀드 값입니다.
    def gte(a, gt):
        if a[0] >= gt[0] and a[1] <= gt[1] and a[2] <= gt[2]:
            return (255, 0, 0)
        return (0, 0, 0)
    im.putdata( list(map(lambda x: gte(x, threshold), im.getdata())) )

def makeBox(type):
    y = []
    cnt = 0

    if type == 10:
        path = fr"{cfg.exePath}img/원-배치-태스크/불량데이터/0"
    elif type == 1:
        path = fr"{cfg.exePath}img/원-배치-태스크/불량데이터/1"
    elif type == 2:
        path = fr"{cfg.exePath}img/원-배치-태스크/불량데이터/2"
    elif type == 3:
        path = fr"{cfg.exePath}img/원-배치-태스크/불량데이터/3"

    for i in range(1, 200):
        try:
            im = Image.open(fr'{path}/m{i}.jpg')
            im = transforms.Resize((224, 224))(im)
            setThresHold(im)
            y.append(knn.knn(imBW=im, type=type))
            cnt = i
        except FileNotFoundError as e:
            break
    return y, cnt, type