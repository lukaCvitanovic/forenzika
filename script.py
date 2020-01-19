import cv2
from matplotlib import pyplot as plt
import numpy as np
import skimage.restoration as restoration
import seminar

def gama_corection(pic, gama):
    new_img = np.uint8(cv2.pow(pic / 255.0, gama) * 255)
    return new_img


# e = 0.5~0.8  w =  0.2~0.5
# T = 128
def determine_gama(img, T = 128, e = 0.65, w = 0.35):
    hist = seminar.createHistogram(None, img, False, False, False)
    B = 0
    D = 0
    for val in hist:
        if val > T:
            B += val
        else:
            D += val

    if B >= D:
        gama = 1 + e
    else:
        gama = 1 - w

    return gama

# pis is gray-scale singel chanel image
def create_subbands_histograms(pic, first = 0, last = 64):
    # dividing image into 8x8 blocks and extracting DCT coeficients
    blocks = []
    coefs = []
    for i in range(64):
        coefs.append([])

    for row in range(0, pic.shape[0], 8):
        for col in range(0, pic.shape[1], 8):
            block = []
            for r_of in range(8):
                brow = []
                for c_of in range(8):
                    brow.append(pic[row + r_of][col + c_of])
                block.append(brow)
            block = np.array(block)

            # getting DCT coef for 8x8
            float = np.float32(block)
            coef = cv2.dct(float)
            # coef = coef.flatten()
            blocks.append(block)
            for i in range(8):
                for j in range(8):
                    coefs[i * 8 + j].append(coef[i][j])

    # create histogram of coef
    for gram in range(first, last):
        tcoefs = np.array(coefs[gram])
        tcoefs.sort()
        tcoefs = np.int32(tcoefs)
        start = tcoefs[0]
        end = tcoefs[-1]
        hist = [0] * (end + 1 + int(start * (-1)))
        for val in tcoefs:
            hist[val + int(start * (-1))] += 1

        rang = np.arange(int(start), end + 1)
        plt.bar(rang, hist, 1)
        plt.title('DCT Coefitients [{}][{}]'.format(gram // 8, gram % 8))
        plt.xlabel('Coef num')
        plt.ylabel('number of coef')
        plt.show()


def double_compresion():
    # load tif image
    img_name = 'ucid00024.tif'
    jpg_path1 = 'ws/image60.jpg'
    jpg_path2 = 'ws/image95.jpg'
    img = cv2.imread(img_name)

    # save tiff image as jpeg with 60% compresion
    cv2.imwrite(jpg_path1, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # load jpg image
    jpg_img = cv2.imread(jpg_path1)

    # save jpeg60 as jpeg95 image
    cv2.imwrite(jpg_path2, jpg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # load jpeg95 and convert it to gray scale
    gray = seminar.convertToGrayScale(jpg_path2, False)

    # reducing data from 3 to 1 chanel
    final = []
    for row in range(gray.shape[0]):
        temp = []
        for col in range(gray.shape[1]):
            temp.append(gray[row][col][0])
        final.append(temp)
    final = np.array(final)

    return final


def main():
    final_path = 'ws/final.jpg'

    img = double_compresion()
    seminar.createHistogram(None, img, False, False, True)
    corected = gama_corection(img, determine_gama(img))
    seminar.createHistogram(None, corected, False, False, True)
    corected = np.float32(corected)
    deblocked = restoration.denoise_tv_bregman(corected, 2)
    deblocked = np.uint8(deblocked)
    seminar.createHistogram(None, deblocked, False, False, True)
    create_subbands_histograms(deblocked, 18, 19)

    cv2.imwrite(final_path, deblocked, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    print('Forged image created')
    print('Image saved at: {}'.format(final_path))

if __name__ == '__main__':
    main()