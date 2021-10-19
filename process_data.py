import cv2
import sys
from tqdm.auto import tqdm
from jax import numpy as jnp
import numpy as np
import jax
from glob import glob
from multiprocessing import cpu_count, Pool
import utils

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
    

def load_data(data_dir):
    """
    Loads all images in a directory to memory in parallel, using the functions above
    """
    # TODO: fix lung_Opacity here
    classes = utils.CLASS_NAMES
    all_images = []
    for cls in classes:
        images = glob(cls + '/*')
        images = parallel_map(read_image, images)
        images = np.array(images)
        all_images.append(images)
    return all_images

resolution = 256

# If running as a script, loads all images from the data directory and
# saves them as "x.npy" and "y.npy" in the current directory
if __name__ == '__main__':
    
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        resolution = int(sys.argv[2])

    foldername = sys.argv[1]
    data = load_data(foldername)
    x = np.concatenate(data)

    y = [np.ones(d.shape[0:1]) * i for i,d in enumerate(data)]
    y = np.concatenate(y)
    
    with open(foldername + '/x.npy', 'wb') as f:
        np.save(f, x)

    with open(foldername + '/y.npy', 'wb') as f:
        np.save(f, y)
