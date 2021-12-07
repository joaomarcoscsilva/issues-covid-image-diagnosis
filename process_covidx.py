import cv2
import sys
from tqdm.auto import tqdm
from jax import numpy as jnp
import numpy as np
import jax
from glob import glob
from multiprocessing import cpu_count, Pool
import pickle

# Because the dataset is very large, set jax to use CPU memory by default
jax.config.update('jax_platform_name', 'cpu')


def parallel_map(fn, inputs):
    """
    Applies the function fn to every element in a list in parallel (on the cpu)
    """

    with Pool(cpu_count()) as pool:
        return list(tqdm(pool.imap(fn, inputs), total = len(inputs)))


@jax.jit
def process_image(im):
    """
    Processes one image, scaling the values to [0,1] and resizing to the correct resolution
    """

    im = jnp.array(im, dtype = jnp.float32)
    im = im / 255
    
    im = jax.image.resize(im, (resolution, resolution, 3), 'nearest')
    return im

def read_image(path):
    """
    Reads the image in the given path and return it after pre-processing.
    """

    im = cv2.imread(path)
    return process_image(im)

def load_labels(label_filename):
    with open(label_filename, 'r') as f:
        lines = f.readlines()
        lines = map(lambda x: x.split(' '), lines)

        labels = dict()

        for l in lines:
            if len(l) == 4:
                labels[l[1]] = l[2]
            else:
                labels[l[2]] = l[3]

        return labels

def load_data(data_dir, labels):
    """
    Loads all images in a directory to memory in parallel, using the functions above
    """
    paths = [data_dir + '/' + img for img in labels.keys()]
    images = parallel_map(read_image, paths)
    images = np.array(images)

    return images, paths

resolution = 256

# If running as a script, loads all images from the data directory and
# saves them as "x.npy" and "y.npy" in the current directory
if __name__ == '__main__':

    y_train = load_labels('covidx/train_COVIDx9A.txt')
    y_test = load_labels('covidx/test_COVIDx9A.txt')

    x_train, train_paths = load_data('covidx/train', y_train)
    x_test, test_paths = load_data('covidx/test', y_test)
    
    with open('covidx/x_train.npy', 'wb') as f:
        np.save(f, x_train)

    with open('covidx/x_test.npy', 'wb') as f:
        np.save(f, x_test)

    d = {'normal' : 0, 'pneumonia' : 1, 'COVID-19' : 2}

    print(np.unique(y_train.values()))

    y_train = list(map(d.get, y_train.values()))
    y_test = list(map(d.get, y_test.values()))

    with open('covidx/y_train.npy', 'wb') as f:
        np.savez(f, y_train)

    with open('covidx/y_test.npy', 'wb') as f:
        np.save(f, y_test)

    with open('covidx/metadata.pickle', 'wb') as f:
        pickle.dump({
            "classnames": ['Normal', 'Pneumonia', 'COVID-19'], "train_paths" : train_paths, "test_paths" : test_paths
        }, f)
