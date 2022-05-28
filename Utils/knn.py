from torchvision import transforms
import random
import numpy as np
'''
knnResult 폴더에 k-NN 결과 빨간 영역으로 crop된 이미지가 저장되고 그 이미지를 토대로,
패치 단위 마이크로 이미지들이 patching.patching() 함수를 통해 생성된다.
'''
def knn(imBW, type, resizeFactor=1):
    n = imBW.size[0]
    n *= resizeFactor
    n = int(n)
    imBW = transforms.Resize((n, n))(imBW).convert('RGB')
    im = imBW.load()
    visitCount = 0
    visit = np.zeros((n, n))

    [rx1, ry1, rx2, ry2] = [9999, 9999, 0, 0]
    for randomSeedingCount in range(5000):
        x = random.randint(0, n)
        y = random.randint(0, n)
        que = [(x, y)]
        adj = [[0,1],[1,0], [0,-1],[-1,0]]
        while que:
            x, y = que.pop()
            if x >= n or y >= n or x < 0 or y < 0:
                continue
            visit[x, y] = 1
            visitCount = visitCount + 1

            if len(que) > 0:
                if rx1 > x:
                    rx1 = x
                if rx2 < x:
                    rx2 = x
                if ry1 > y:
                    ry1 = y
                if ry2 < y:
                    ry2 = y

            for j in range(0, 4):
                xx, yy = adj[j]
                if x + xx < n and y + yy < n:
                    if x + xx > 0 and y + yy > 0:
                        if not visit[x + xx, y + yy] and im[x + xx, y + yy] >= (230, 0, 0):
                            im[x + xx, y + yy] = (255, 0, 0)
                            que.append((x + xx, y + yy))

        #imArr[i].show()

    imBW = transforms.Resize((108, 108))(imBW)
    #print(f'img/patch/진주황-빛반사-불량/kNN/u0.jpg')
    #imBW.save(f'img/patch/진주황-빛반사-불량/kNN/u0.jpg')
    #print(len(imArr[0].getdata()) // 112)
    scaleFactor = 1/resizeFactor

    def round2(f):
        return round(f, 2)

    #print(f'crop={(rx1, ry1, rx2, ry2)}')
    #print(f"idx=는 knn으로 간다. crop={(rx1*scaleFactor, ry1*scaleFactor, rx2*scaleFactor, ry2*scaleFactor)}")

    if type is not 10:
        type = 0

    resArr = [
    round2(rx1*scaleFactor / 224 * 11),
    round2(ry1*scaleFactor / 224 * 11),
    round2(rx2*scaleFactor / 224 * 11),
    round2(ry2*scaleFactor / 224 * 11),
        type
    ]
    print(f"{resArr},")
    return resArr
