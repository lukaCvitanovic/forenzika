import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

def importPic(path,display,mode):
    if isinstance(mode,int):
        pic = cv2.imread(path,mode)
    elif mode is None:
        pic = cv2.imread(path)
    if display:
        show(pic, "slika u boji")
    return pic


# format u kojem su spremljene boje u slikama su BGR
def convertToGrayScale(path,display):
    pic = importPic(path, False,None)
    for redak in pic:
        for stupac in redak:
            value = int(stupac[0]*0.114 + stupac[1]*0.587 + stupac[2]*0.299)
            for boja in range(0,3):
                stupac[boja] = value
    if display:
        show(pic,"slika")
    return pic

# samo za gray-scale slike
def createHistogram(path, pic, cumulative, show_pic,show_hist):
    if path is not None:
        pic = importPic(path,show_pic,0)
    hist = [0] * 256
    for redak in pic:
        for stupac in redak:
            hist[int(stupac)] += 1
    if cumulative:
        for value in range(0,256):
            if value != 0:
                hist[value] += hist[value-1]
    if show_hist:
        arr = np.array(hist)
        plt.bar(np.arange(len(arr)),arr,1)
        if cumulative:
            plt.title('Kumulativni histogram slike')
        else:
            plt.title('Histogram slike')
        plt.xlabel('Intenziteti sive boje')
        plt.ylabel('Broj piksela')
        plt.show()
    return hist

# napravi da za min u maksimum uzima 5% od srednje vrijednosti ili mediana
def normalizeHistogram(path, display):
    pic = importPic(path, False,0)
    hist = createHistogram(path, None, False, True,True)
    min = 0
    for value in hist:
        if value == 0:
            min += 1
        else:
            break
    max = 255
    for value in list(reversed(hist)):
        if value == 0:
            max -= 1
        else:
            break
    for redak in range(0,len(pic)):
        for stupac in range(0,len(pic[0])):
                pic[redak][stupac] = int((pic[redak][stupac]-min)*(255/(max-min)))
    hist2 = createHistogram(None,pic,False,True,True)
    if display:
        show(pic,"slika")

def equalizeHistogram2(path):
    pic = importPic(path,False,0)
    hist = createHistogram(None,pic,False,True,True)
    cum_hist = createHistogram(None,pic,True,False,True)
    for value in range(0, 256):
        hist[value] /= cum_hist[255]
    cdf = hist.copy()
    for value in range(0, 256):
        if value != 0:
            cdf[value] += cdf[value - 1]
    new_hist = cdf.copy()
    for value in range(0, 256):
       new_hist[value] = int(cdf[value] * 255)
    for redak in range(0,len(pic)):
        for stupac in range(0,len(pic[0])):
                pic[redak][stupac] = new_hist[pic[redak][stupac]]
    createHistogram(None,pic,False,True,True)
    show(pic, "izjednacen histogram")

def gammaCorrection(path, correction):
    gray = convertToGrayScale(path,True)
    corr = np.uint8(cv2.pow(gray/255.0, correction)*255)
    show(corr, "Gamma korekcija")

def amplitudeSegmetation(path,img,border=0):
    if img:
        pic = path
    else:
        pic = importPic(path,False,0)
    if border == 0:
        createHistogram(None, pic, False, True, True)
        print("unesite razinu amplituden segmentacije (0-255)")
        border = input()
    for redak in range(0,pic.shape[0]):
        for stupac in range(0,pic.shape[1]):
            if pic[redak][stupac] <= int(border):
                pic[redak][stupac] = 0
            else:
                pic[redak][stupac] = 255
    show(pic, "Amplitudna segmentacija oko vrijednosti" + str(border))
    return pic

def corelation(path,karnel):
    pic = importPic(path,True,0)
    new_img = conv_cor(pic,False,karnel,False,False,False)
    new_pic = np.array(new_img)
    show(new_pic,"korelacija")

# shape = (redci,stupci)
def convolition(path, median, karnel):
    pic = importPic(path,True,0)
    if median:
        karnel = [[1, 1, 1], [1, '1?', 1], [1, 1, 1]]
        new_img = conv_cor(pic, True, karnel, True, False,False)
    else:
        new_img = conv_cor(pic,True,karnel,False,False,False)
    new_pic = np.array(new_img)
    show(new_pic,"konvolucija")
    return new_pic

def imageSharpening(path,karnel):
    edges = convolition(path,False,karnel)
    pic = importPic(path,False,0)
    sharpen = combineImages(edges,pic,True,False,0.5)
    show(sharpen,"Sharpen image")
    #cv2.imwrite('C:\\Users\\Luka\\Desktop\\shapren.jpg',np.array(sharpen))
    #cv2.imwrite('C:\\Users\\Luka\\Desktop\\gray.jpg',pic)

# radi za slike istih dimenzija
def combineImages(path1, path2, img, edge, alfa):
    if img:
        pic1 = path1
        pic2 = path2
    else:
        pic1 = importPic(path1,True,0)
        pic2 = importPic(path2,True,0)
    if (len(pic1) != len(pic2)) or (len(pic1[0]) != len(pic2[0])):
        if (len(pic1)*len(pic1[0])) > (len(pic2)*len(pic2[0])):
            pic2 = cv2.resize(pic2,(len(pic1),len(pic1[0])))
        else:
            pic1 = cv2.resize(pic1, (len(pic2), len(pic2[0])))
    new_img = []
    if edge:
        angle = []
    for redak in range(0,len(pic1)):
        new_img.append(pic1[redak].copy())
        if edge:
            angle.append(pic1[redak].copy())
    for redak in range(0,len(pic1)):
        for stupac in range(0,len(pic1[0])):
            if edge:
                a = math.pow(pic2[redak][stupac],2)
                b = math.pow(pic1[redak][stupac],2)
                new_img[redak][stupac] = math.sqrt(a + b)
                c = math.atan2(pic1[redak][stupac], pic2[redak][stupac]) #+ math.pi
                c *=  180/math.pi
                #if c > 0:
                    #print("i")
                angle[redak][stupac] = c#float(math.atan2((pic1[redak][stupac]), (pic2[redak][stupac])) * math.pi)
            else:
                new_img[redak][stupac] = alfa*pic1[redak][stupac] + (1-alfa)*pic2[redak][stupac]
    if edge:
        return {"pic":new_img, "angle":angle}
    else:
        return new_img

def coloringAngles(angle):
    img = []
    for redak in range(0,len(angle)):
        row = []
        for stupac in range(0,len(angle[0])):
            row.append([angle[redak][stupac],angle[redak][stupac],angle[redak][stupac]])
        img.append(row.copy())
    for r in range(0,len(angle)):
        for s in range(0,len(angle[0])):
            if angle[r][s] < 0:
                angle[r][s] += 360
    for redak in range(0,len(angle)):
        for stupac in range(0,len(angle[0])):
            #img[redak][stupac][0] = 256 * (math.cos(angle[redak][stupac] - 120))
            #img[redak][stupac][1] = 256 * (math.cos(angle[redak][stupac] + 120))
            #img[redak][stupac][2] = 256 * (math.cos(angle[redak][stupac]))
            boje = angle2color(angle[redak][stupac])
            img[redak][stupac][0] = boje["b"]
            img[redak][stupac][1] = boje["g"]
            img[redak][stupac][2] = boje["r"]
    return img

def angle2color(ang):
    if ang == 0:
        return {"r":0,"g":0,"b":0}
    elif ((ang < 22.5) or (ang >= 337.5)) or ((ang < 202.5) and (ang >= 157.5)):
        return {"r":0,"g":255,"b":0}
    elif ((ang < 67.5) and (ang >= 22.5)) or ((ang < 247.5) and (ang >= 202.5)):
        return {"r":255,"g":0,"b":0}
    elif ((ang < 112.5) and (ang >= 67.5)) or ((ang < 292.5) and (ang >= 247.5)):
        return {"r":0,"g":0,"b":255}
    elif ((ang < 157.5) and (ang >= 112.5)) or ((ang < 337.5) and (ang >= 292.5)):
        return {"r":255,"g":255,"b":0}

def hsv2rgb(h):
    v = 1
    s = 1
    c = 1
    x = 1 - abs(math.fmod(h/60,2)-1)
    m = 0
    r1 = 0
    g1 = 0
    b1 = 0
    if (h >= 0) and (h <60):
        r1 = c
        g1 = x
    elif (h >= 60) and (h <120):
        r1 = x
        g1 = c
    elif (h >= 120) and (h <180):
        g1 = c
        b1 = x
    elif (h > 180) and (h <240):
        g1 = x
        b1 = c
    elif (h >= 240) and (h <300):
        r1 = x
        b1 = c
    elif (h >= 300) and (h <360):
        r1 = c
        b1 = x
    elif h == 180:
        return {"r":0,"g":0,"b":0}
    return {"r":r1*255,"g":g1*255,"b":b1*255}

def nonMaximumSuoresion(G,ang):
    grad = []
    for redak in range(0,len(G)):
        grad.append(G[redak].copy())
    for redak in range(0,len(G)):
        for stupac in range(0,len(G[0])):
            if redak == 0 or redak == len(G)-1 or stupac == 0 or stupac == len(G[0])-1:
                grad[redak][stupac] = 0
                continue
            if ((ang[redak][stupac] < 22.5) or (ang[redak][stupac] >= 337.5)) or ((ang[redak][stupac] < 202.5) and (ang[redak][stupac] >= 157.5)):
                if G[redak][stupac] <= G[redak][stupac-1] or G[redak][stupac] <= G[redak][stupac]:
                    grad[redak][stupac] = 0
            elif ((ang[redak][stupac] < 67.5) and (ang[redak][stupac] >= 22.5)) or ((ang[redak][stupac] < 247.5) and (ang[redak][stupac] >= 202.5)):
                if G[redak][stupac] <= G[redak - 1][stupac + 1] or G[redak][stupac] <= G[redak + 1][ stupac - 1]:
                    grad[redak][stupac] = 0
            elif ((ang[redak][stupac] < 112.5) and (ang[redak][stupac] >= 67.5)) or ((ang[redak][stupac] < 292.5) and (ang[redak][stupac] >= 247.5)):
                if G[redak][stupac] <= G[redak - 1][ stupac] or G[redak][ stupac] <= G[redak + 1][ stupac]:
                    grad[redak][stupac] = 0
            elif ((ang[redak][stupac] < 157.5) and (ang[redak][stupac] >= 112.5)) or ((ang[redak][stupac] < 337.5) and (ang[redak][stupac] >= 292.5)):
                if G[redak][stupac] <= G[redak - 1][ stupac - 1] or G[redak][stupac] <= G[redak + 1][stupac + 1]:
                    grad[redak][stupac] = 0
    return grad

def hysteresis(strongEdges,thresholdedEdges,pic):
    finalEdges = strongEdges.copy()
    currentPixels = []
    for r in range(1, pic.shape[0] - 1):
        for c in range(1, pic.shape[1] - 1):
            if thresholdedEdges[r, c] != 1:
                continue  # Not a weak pixel

            # Get 3x3 patch
            localPatch = thresholdedEdges[r - 1:r + 2, c - 1:c + 2]
            patchMax = localPatch.max()
            if patchMax == 2:
                currentPixels.append((r, c))
                finalEdges[r, c] = 1

    # Extend strong edges based on current pixels
    while len(currentPixels) > 0:
        newPix = []
        for r, c in currentPixels:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0: continue
                    r2 = r + dr
                    c2 = c + dc
                    if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
                        # Copy this weak pixel to final result
                        newPix.append((r2, c2))
                        finalEdges[r2, c2] = 1
        currentPixels = newPix

    return finalEdges

def upper(pic):
    for r in range(0,len(pic)):
        for s in range(0,len(pic[0])):
            pic[r][s] *= 255
    return pic

def edgeDetection(path):
    pic = importPic(path,True,0)
    gausian = [[0.111111, 0.111111, 0.111111], [0.111111, '0.111111?', 0.111111], [0.111111, 0.111111, 0.111111]]
    blur = conv_cor(pic,True,gausian,False,False,False)
    sobelX = [[-1, 0, 1], [-2, '0?', 2], [-1, 0, 1]]
    sobelY = [[1, 2, 1], [0, '0?', 0], [-1, -2, -1]]
    Gx = conv_cor(np.array(blur),True,sobelX,False,True,True)
    Gy = conv_cor(np.array(blur),True,sobelY,False,True,True)
    #show(np.array(Gx),"Gx")
    #show(np.array(Gy), "Gy")
    dict = combineImages(Gx,Gy,True,True,0)
    #grad = nonMaximumSuoresion(dict["pic"],dict["angle"])
    grad = np.array(dict["pic"],dtype=float)
    custk = [[0, 0, 0], [0, '0?', 0], [0, 0, 0]]
    pos1 = calcDistToEdge(custk)
    ex = np.zeros([(grad.shape[0] + (len(custk) - 1)), (grad.shape[1] + (len(custk) - 1))], float)
    grad = imprintMatrix(ex,grad,pos1)
                  #hightreshold
    strongEdg = np.array((grad>91), dtype=np.uint8)
                           #lowtreshold
    tresholdEdg = strongEdg + (grad>31)
    final = hysteresis(strongEdg,tresholdEdg,pic)
    #show(np.array(dict["pic"]),"G")
    final = upper(final)
    final = cutOriginalImage(final,pos1,dict["pic"])
    Gx = conv_cor(final, True, sobelX.copy(), False, True, False)
    Gy = conv_cor(final, True, sobelY.copy(), False, True, False)
    dict = combineImages(Gx, Gy, True, True,0)
    color = coloringAngles(dict["angle"])
    color = np.array(color)
    show(np.array(dict["pic"]),"Gradient")
    show(color, "magnitude orientation")
    cv2.imwrite('C:\\Users\\Luka\\Desktop\\e.jpg',color)

def conv_cor(pic, conv_cor, jezgra, median, edge, second):
    if jezgra is None:
        karnel = createKernel()
    else:
        karnel = []
        for r in range(0,len(jezgra)):
            karnel.append(jezgra[r].copy())
    #karnel = [[0.111111, 0.111111, 0.111111], [0.111111, '0.111111?', 0.111111], [0.111111, 0.111111, 0.111111]]
    pos = calcDistToEdge(karnel)
    karnel = karnelFinal(karnel, pos)
    if conv_cor:
        karnel = rotate180(karnel)
    extd = np.zeros([(pic.shape[0] + (len(karnel) - 1)), (pic.shape[1] + (len(karnel) - 1))], float)
    extd = imprintMatrix(extd, pic, pos)
    # a = np.array(extd)
    if median:
        extd = matrixMultiMedian(karnel,extd,pos)
    else:
        extd = matrixMulti(karnel, extd, pos, edge, second)
    a = np.array(extd)
    new_img = cutOriginalImage(extd, pos, pic)
    b = np.array(new_img)
    return new_img

def cutOriginalImage(extd,pos,pic):
    original = np.array(pic,dtype=float)
    for redak in range(0,len(extd)):
        if ((redak - pos[0]) >= 0) and ((redak - pos[0]) < len(pic)):
            for stupac in range(0,len(extd[0])):
                if ((stupac - pos[1]) >= 0) and ((stupac - pos[1]) < len(pic[0])):
                    original[redak - pos[0]][stupac - pos[1]] = extd[redak][stupac]
    return original

def median(niz):
    niz.sort()
    if len(niz)%2 == 0:
        return (niz[(int(len(niz)/2) + (len(niz)%2)) -1] + niz[(int(len(niz)/2) + (len(niz)%2))])/2
    else:
        return niz[(int(len(niz)/2)+(len(niz)%2))-1]

def matrixMultiMedian(karnel, extd, pos):
    for redak in range(0,len(extd)-(len(karnel)-1)):
        for stupac in range(0,len(extd[0])-(len(karnel)-1)):
            #množenje karnela i extd
            sum = []
            for redakK in range(0,len(karnel)):
                for stupacK in range(0,len(karnel)):
                    sum.append(extd[redak+redakK][stupac+stupacK])
            extd[redak + pos[0]][stupac + pos[3]] = median(sum)
    return extd

def matrixMulti(karnel, extd, pos, sobel, second):
    new_image = []
    for redak in range(0,len(extd)):
        new_image.append(extd[redak].copy())
    for redak in range(0,len(extd)-(len(karnel)-1)):
        for stupac in range(0,len(extd[0])-(len(karnel)-1)):
            #množenje karnela i extd
            sum = 0
            for redakK in range(0,len(karnel)):
                for stupacK in range(0,len(karnel)):
                    sum += ((extd[redak + redakK][stupac + stupacK]) * (karnel[redakK][stupacK])) / 255
            if sum < 0:
                if sobel:
                    if second:
                        sum = abs(sum)
                else:
                    sum = 0
            elif sum > 1:
                sum = 1
            sum *= 255
            new_image[redak+pos[0]][stupac+pos[3]] = sum
    return new_image

def rotate180(karnel):
    temp = []
    for row in range(0,len(karnel)):
        temp.append(karnel[row].copy())
    for redak in range(0,len(karnel)):
        for stupac in range(0,len(karnel)):
            temp[redak][stupac] = karnel[(len(karnel)-1)-redak][(len(karnel[0])-1)-stupac]
    return temp

def karnelFinal(karnel, pos):
    chars = list(karnel[pos[0]][pos[3]])
    string = []
    for char in range(0,len(chars)-1):
        string.append(chars[char])
    string = ''.join(string)
    num = float(string)
    karnel[pos[0]][pos[3]] = num
    return karnel

def Max(lista):
    max = lista[0]
    for el in lista:
        if el > max:
            max = el
    return max

def imprintMatrix(exdt,pic,pos):
    for redak in range(0,len(exdt)):
        if ((redak - pos[0]) >= 0) and ((redak - pos[0]) < pic.shape[0]):
            for stupac in range(0,len(exdt[0])):
                if ((stupac - pos[3]) >= 0) and ((stupac - pos[3]) < pic.shape[1]):
                    exdt[redak][stupac] = pic[redak - pos[0]][stupac - pos[3]]
    return exdt

def calcDistToEdge(kernel):
    for redak in range(0,len(kernel)):
        for stupac in range(0,len(kernel[redak])):
            if isinstance(kernel[redak][stupac],str):
                return [redak,(len(kernel)-1)-stupac,(len(kernel)-1)-redak,stupac]
                       #gori  desno                  doli                 livo  (pozicija pikesela na kojeg zbrajamo u karnelu)
    return None

# kernele se unosi redak po redak
# kernel mora biti kvadratna matrica
# element na koji se kerenel primjenjuje je oznaćen sa sa sufiksom ? (2?)
def createKernel():
    print('input kernel row 1')
    red = input()
    red = createKernelRow(red)
    kernel = list()
    kernel.append(red)
    for turn in range(0,len(red)-1):
        print('input kernel row '+str(turn + 2))
        temp = input()
        temp = createKernelRow(temp)
        kernel.append(temp)
    return kernel

# pretvara string u niz brojeva
def createKernelRow(row):
    chars = list(appendSpace(row))
    nums = list()
    start = 0
    for char in range(0,len(chars)):
        if chars[char] == ' ':
            num = list()
            for val in range(start,char):
                num.append(chars[val])
            num = ''.join(num)
            if chars[char - 1] != '?':
                num = float(num)
            nums.append(num)
            start = char+1
    return nums

def appendSpace(Row):
    chars = list(Row)
    chars.append(' ')
    return ''.join(chars)

#binary image only
def erosion_dialation(path,dim,shw,erosion):
    if dim % 2 == 1:
        if isinstance(path,str):
            pic = importPic(path,True,0)
        else:
            pic = path
        binary = amplitudeSegmetation(pic,True)
        row = [1] * dim
        karnel = []
        for r in range(0,dim):
            karnel.append(row.copy())
        karnel[int(dim / 2)][int(dim / 2)] = '1?'
        pos = calcDistToEdge(karnel)
        karnel = karnelFinal(karnel,pos)
        if erosion:
            extd = np.zeros([(len(binary) + len(karnel)-1), (len(binary[0]) + len(karnel)-1)], float)
        else:
            extd = np.ones([(len(binary) + len(karnel)-1), (len(binary[0]) + len(karnel)-1)], float)
        extd = imprintMatrix(extd,binary,pos)
        extd = erode_dilate(extd,karnel,pos,erosion)
        final = cutOriginalImage(extd,pos,binary)
        if shw:
            show(final,"Eroded image")
            #cv2.imwrite('C:\\Users\\Luka\\Desktop\\e.jpg',final)
        return final

def erode_dilate(bin,karnel,pos,erode):
    new_image = bin.copy()
    for redak in range(0, len(bin) - (len(karnel) - 1)):
        for stupac in range(0, len(bin[0]) - (len(karnel) - 1)):
            # množenje karnela i extd
            if erode:
                sum = 1
            else:
                sum = 0
            for redakK in range(0, len(karnel)):
                for stupacK in range(0, len(karnel)):
                    temp = (bin[redak + redakK][stupac + stupacK]) * (karnel[redakK][stupacK])
                    if erode:
                        sum *= temp
                    else:
                        sum += temp
            if sum > 0:
                new_image[redak + pos[0]][stupac + pos[3]] = 255
            elif sum == 0:
                new_image[redak + pos[0]][stupac + pos[3]] = 0
    return new_image

def opening_closing(path,opening,dim):
    if opening:
        erode = erosion_dialation(path,dim,True,True)
        erosion_dialation(erode,dim,True,False)
    else:
        dilate = erosion_dialation(path, dim, True, False)
        erosion_dialation(dilate, dim, True, True)

def show(image, window_name):
    pic = np.array(image,dtype=np.uint8)
    cv2.imshow(window_name, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot(pic, cumulative):
    plt.hist(pic.ravel(), 256, [0, 256], cumulative=cumulative)
    plt.title('Histogram slike')
    plt.xlabel('Intenziteti sive boje')
    plt.ylabel('Broj piksela')
    plt.show()
    plt.close()

if __name__ == '__main__':
    #1) Transformacija slike u boji u sliku sivih razina
    #convertToGrayScale('C:\\Users\\Luka\\Desktop\\b.jpg',True)
    #2) Računanje histograma i kumulativnog histograma
    #createHistogram('C:\\Users\\Luka\\Desktop\\b.jpg',None,False,True,True)
    #createHistogram('C:\\Users\\Luka\\Desktop\\b.jpg', None, True, False, True)
    #3) Rastezanje histograma
    #normalizeHistogram('C:\\Users\\Luka\\Desktop\\h.jpg',True)
    #4) Ujednačavanje histograma
    #equalizeHistogram2('C:\\Users\\Luka\\Desktop\\b.jpg')
    #5) Gamma korekcija
    #gammaCorrection('C:\\Users\\Luka\\Desktop\\b.jpg',2.5)
    #6) Amplitudna segmentacija (korisnik odabire prag temeljem prikazanog histograma slike)
    #amplitudeSegmetation('C:\\Users\\Luka\\Desktop\\b.jpg',False)
    #7) Konvolucija (proizvoljnim filtrom koji zadaje korisnik)
    #a = [[0.111111, 0.111111, 0.111111], [0.111111, '0.111111?', 0.111111], [0.111111, 0.111111, 0.111111]]
    #convolition('C:\\Users\\Luka\\Desktop\\b.jpg',False,a)
    #8) Korelacija (proizvoljnim uzorkom koji zadaje korisnik)
    #a = [[0.111111, 0.111111, 0.111111], [0.111111, '0.111111?', 0.111111], [0.111111, 0.111111, 0.111111]]
    #corelation('C:\\Users\\Luka\\Desktop\\b.jpg',a)
    #9) Median filter
    #convolition('C:\\Users\\Luka\\Desktop\\b.jpg',True,[])
    #10) Image sharpening
    #a = [[0,-1,0],[-1,'4?',-1],[0,-1,0]]
    #imageSharpening('C:\\Users\\Luka\\Desktop\\b.jpg',a)
    #11) Otkrivanje magnitude i orjentacije rubova na slici
    #edgeDetection('C:\\Users\\Luka\\Desktop\\g.jpg')
    #12) Kombiniranje slika
    #comb = combineImages('C:\\Users\\Luka\\Desktop\\g.jpg','C:\\Users\\Luka\\Desktop\\b.jpg',False,False,0.5)
    #show(comb,"Combine image")
    #13) Morfološke operacije na binarnoj slici (erozija, dilatacija, opening, closing)
    # erozia
    #erosion_dialation('C:\\Users\\Luka\\Desktop\\bef.jpg',3,True,True)
    # dilatacija
    #erosion_dialation('C:\\Users\\Luka\\Desktop\\bef.jpg', 3, True, False)
    # otvaranje
    #opening_closing('C:\\Users\\Luka\\Desktop\\bef.jpg',True,3)
    # zatvaranje
    #opening_closing('C:\\Users\\Luka\\Desktop\\bef.jpg', False, 3)