#coding: utf-8

from PIL import Image
import numpy as np
import feature
import pickle

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