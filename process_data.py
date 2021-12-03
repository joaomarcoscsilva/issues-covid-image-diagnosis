import pickle
import cv2
import sys
from tqdm.auto import tqdm
from jax import numpy as jnp
import numpy as np
import jax
from glob import glob
from multiprocessing import cpu_count, Pool

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

preferred_class_order = {
    'normal': 0,
    'pneumonia viral': 1,
    'viral pneumonia': 1,
    'covid': 2,
    # 3 is either lung opacity or bact pneumonia
}

preferred_names = [
    'Normal',
    'Viral pneumonia',
    'COVID-19'
]

def load_data(data_dir):
    """
    Loads all images in a directory to memory in parallel, using the functions above
    """
    classes = glob(data_dir + '/*/')
    all_images = []

    classes_ordered = []
    classes_not_added = []

    for cls in classes:
        formatted_cls = cls.split('/')[-2].lower().replace('_', ' ').replace('-', ' ').replace('19', '').strip()
        
        if formatted_cls in preferred_class_order:
            idx = preferred_class_order[formatted_cls]
            classes_ordered.insert(idx, { 'foldername': cls, 'classname': preferred_names[idx] })
        else:
            classes_not_added.append({ 'foldername': cls, 'classname': formatted_cls.capitalize() })

    for cls in classes_not_added:
        classes_ordered.append(cls)

    if len(classes_not_added) > 1:
        print("WARNING: MORE THAN ONE CLASS WITHOUT SPECIFIC ORDERING. ABORTING...")
        print(classes_not_added)
        return
    
    print("CLASS ORDER: ", classes_ordered)

    for cls in classes_ordered:
        images = glob(cls['foldername'] + '/*')
        images = parallel_map(read_image, images)
        images = np.array(images)
        all_images.append(images)
    
    ret_classnames = []
    for cls in classes_ordered:
        ret_classnames.append(cls['classname'])

    return all_images, ret_classnames

resolution = 256

# If running as a script, loads all images from the data directory and
# saves them as "x.npy" and "y.npy" in the current directory
if __name__ == '__main__':
    
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        resolution = int(sys.argv[2])

    foldername = sys.argv[1]
    data, class_names = load_data(foldername)
    x = np.concatenate(data)

    y = [np.ones(d.shape[0:1]) * i for i,d in enumerate(data)]
    y = np.concatenate(y)
    
    with open(foldername + '/x.npy', 'wb') as f:
        np.save(f, x)

    with open(foldername + '/y.npy', 'wb') as f:
        np.save(f, y)
    
    with open(foldername + '/metadata.pickle', 'wb') as f:
        pickle.dump({
            "classnames": class_names
        }, f)