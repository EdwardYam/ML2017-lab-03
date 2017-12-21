#coding: utf-8

from PIL import Image
import numpy as np
import feature
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

def read_image(filename):
    im = Image.open(filename)
    im = im.convert('L')
    return im

def trans_to_array(image, size=None):
    if(size == None):
        size = image.size
    image = image.resize(size)
    return np.asarray(image)


def read_files(dir, count, is_face):
    array = []
    for i in range(0, count):
        s = "%03d" % i
        if(is_face):
            filename = dir + "face_%s.jpg" % s
        else:
            filename = dir + "nonface_%s.jpg" % s
        #print(filename)
        im = read_image(filename)
        im_array = trans_to_array(im, (24, 24))
        ndp_feature = feature.NPDFeature(im_array)
        result = ndp_feature.extract()
        array.append(result)

    array = np.array(array)
    if (is_face):
        y = np.ones((array.shape[0], 1))
    else:
        y = -1 * np.ones((array.shape[0], 1))

    array = np.concatenate((array, y), axis=1)
    #print(array[:, -1])
    if (is_face):
        with open('face.data', 'wb') as file:
            pickle.dump(array, file)
    else:
        with open('nonface.data', 'wb') as file:
            pickle.dump(array, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    #print(data[:][-1])
    return data


def process_data():
    # preprocess data
    read_files("datasets/original/face/", 500, True)
    read_files("datasets/original/nonface/", 500, False)


if __name__ == "__main__":
    process_data()
    # x = np.array(
    #     [1, 2, 5, 10]p
    # )

    # x_1 = np.array(
    #     [1]
    # )
    #
    # x_2 = np.array(
    #     [1, 2]
    # )
    #
    # x_5 = np.array(
    #     [1, 2, 3, 4, 5]
    # )
    #
    # x_10 = np.array(
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # )
    #
    # err_1 = np.array(
    #     [0.10000000000000001]
    # )
    #
    # err_2 = np.array(
    #     [0.10000000000000001, 0.10625000000000002]
    # )
    #
    # err_5 = np.array(
    #     [0.10000000000000001, 0.10625000000000002, 0.095228301110654057, 0.092713246727486168, 0.0841296046044087]
    # )
    #
    # err_10 = np.array(
    #     [0.10000000000000001, 0.10625000000000002, 0.095228301110654057, 0.092713246727486168, 0.0841296046044087,
    #      0.11548585274878526, 0.11192876955525401, 0.10537065766474661, 0.14233048985345381, 0.1526632836584533]
    # )
    #
    #
    # p = np.array(
    #     [0.83, 0.83, 0.92, 0.93]
    # )
    #
    # r = np.array(
    #     [0.83, 0.83, 0.92, 0.93]
    # )
    #
    # f = np.array(
    #     [0.83, 0.83, 0.92, 0.93]
    # )
    #
    # plot_x = np.linspace(1, 4, 4)
    #
    # plt.figure(1)
    # #plt.plot(x, p, label="Precision")
    # #plt.plot(x, r, label="Recall")
    # #plt.plot(x, f, label="F1 Score")
    #
    # # plt.plot(x_1, err_1, label="1 base leaner")
    # # plt.plot(x_2, err_2, label="2 base leaners")
    # # plt.plot(x_5, err_5, label="5 base leaners")
    # plt.plot(x_10, err_10)
    #
    # plt.legend(loc="upper left")
    # plt.xlabel('Number of base learners')
    # plt.ylabel('Error rate')
    # plt.show()