# ucid00024

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import seminar


def main():
    # load tif image
    img_name = 'ucid00024.tif'
    jpg_path1 = 'ws/image60.jpg'
    jpg_path2 = 'ws/image95.jpg'
    img = cv2.imread(img_name)
    #img = Image.open(img_name)

    # show image
    # cv2.imshow('TIFF image', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # save tiff image as jpeg with 60% compresion
    cv2.imwrite(jpg_path1, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    #img.save(jpg_path1, quality=100)

    # load jpg image
    jpg_img = cv2.imread(jpg_path1)
    #jpg_img = Image.open(jpg_path1)

    # show jpg image
    # cv2.imshow('JPEG 60 image', jpg_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # save jpeg60 as jpeg95 image
    cv2.imwrite(jpg_path2, jpg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    #jpg_img.save(jpg_path2, quality=40)

    # load jpeg95 and convert it to gray scale
    gray = seminar.convertToGrayScale(jpg_path2, False)
    # jpg_img2 = Image.open(jpg_path2)
    # jpg_img2 = np.array(jpg_img2.getdata()).reshape(jpg_img2.size[0], jpg_img2.size[1], 3)
    #
    # for redak in jpg_img2:
    #     for stupac in redak:
    #         value = int(stupac[0]*0.114 + stupac[1]*0.587 + stupac[2]*0.299)
    #         for boja in range(0,3):
    #             stupac[boja] = value

    # reducing data from 3 to 1 chanel
    final = []
    for row in range(gray.shape[0]):
        temp = []
        for col in range(gray.shape[1]):
            temp.append(gray[row][col][0])
        final.append(temp)
    final = np.array(final)

    # dividing image into 8x8 blocks and extracting DCT coeficients
    blocks = []
    coefs = []
    for i in range(64):
        coefs.append([])

    for row in range(0, final.shape[0], 8):
        for col in range(0, final.shape[1], 8):
            block = []
            for r_of in range(8):
                brow = []
                for c_of in range(8):
                    brow.append(final[row + r_of][col + c_of])
                block.append(brow)
            block = np.array(block)

            # getting DCT coef for 8x8
            float = np.float32(block)
            coef = cv2.dct(float)
            #coef = coef.flatten()
            blocks.append(block)
            for i in range(8):
                for j in range(8):
                        coefs[i*8 + j].append(coef[i][j])

    # create histogram of coef
    for gram in range(18,64):
        tcoefs = np.array(coefs[gram])
        tcoefs.sort()
        tcoefs = np.int32(tcoefs)
        start = tcoefs[0]
        end = tcoefs[-1]
        hist = [0] * (end + 1 + int(start*(-1)))
        for val in tcoefs:
            hist[val + int(start*(-1))] += 1

        # diplaying coefs
        #plt.hist(coef, range=(-200, 200))
        rang = np.arange(int(start), end + 1)
        plt.bar(rang, hist, 1)
        plt.title('DCT Coefitients [{}][{}]'.format(gram//8, gram%8))
        plt.xlabel('Coef num')
        plt.ylabel('number of coef')
        plt.show()

    a = 5

    #hist = seminar.createHistogram(None, img, False, True, True)


if __name__ == '__main__':
    main()