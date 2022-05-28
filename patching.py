import random
from PIL import Image, ImageDraw
from Utils.cfg import exePath

def patching(idx, imjpg):
    imjpg = imjpg.resize((800,800))
    im = Image.new('RGBA', imjpg.size, (255, 255, 255, 255))
    im.paste(imjpg)

    def crop(im2, x1, y1, x2, y2):
        return im2.crop((x1, y1, x2, y2))
    def makeRatio(w,h):
        class Min():
            def __init__(self):
                self.w = 2
                self.h = 2
                self.v = 999
        min = Min()
        arr = [[2,2],[2,1],[1,2],[1,1]]
        arr = [[3,3]]
        for i in range(len(arr)):
            wi, hi = arr[i]
            n = wi / hi - w / h
            n = n if n > 0 else -n
            if min.v > n:
                min.v = n
                min.w = wi
                min.h = hi
                print(f'차이={wi / hi - w / h}')
        print(f"선택된 에스펙트 레이시오: {min.w}, {min.h}")
        return min.w, min.h

    ws, hs = im.size
    wn, hn = makeRatio(ws, hs)

    w = []
    h = []
    for i in range(0, wn+1):
        w.append(ws * i // wn)
    for i in range(0, hn+1):
        h.append(hs * i // wn)

    print(w, h)
    for k in range(0, 2):
        rad = random.randint( k * 10, (k + 1) * 10 )
        rad = (k * 15 + (k + 1) * 15) // 2
        rad = 0 if k == 0 else rad
        im = im if k == 0 else im.transform(im.size, Image.AFFINE, (1, 0, -50, 0, 1, 0))

        dst = Image.new('RGBA', (im.width+10, im.height+10), (255, 255, 255, 255))
        print(f'dst.size={dst.size}')
        draw = ImageDraw.Draw(dst)

        imw = im.width // wn
        imh = im.height // hn
        def paste(im, xi, yi):
            #print((imw * xi, imh * yi))
            dst.paste(im, (imw * xi, imh * yi))
            draw.rectangle(((imw * xi, imh * yi), (imw * (xi + 1), imh * (yi + 1))), outline=(255, 0, 101), width=2)

        for i in range(wn):
            for j in range(hn):
                im2 = crop(im.rotate(rad), w[i],h[j],w[i+1],h[j+1])
                paste(im2.copy(), i, j)
                print(i, j)

                background = Image.open(fr"{exePath}img/원-배치-태스크/불량데이터/배경/u1.jpg").resize(im2.size)
                background.paste(im2, mask=im2.split()[3])
                background.save(fr'{exePath}img/원-배치-태스크/[idx{idx}][{rad}rad]{i}.{j}.jpg')
