import jax
from jax import numpy as np
from glob import glob


def load_data(path, rng, test_size = 0.1, num_classes = 4):
    """
    Loads the dataset to memory, already shuffled and split into train and test.
    """

    x = jax.device_put(np.load(path + '/x.npy'), jax.devices('cpu')[0])
    y = jax.device_put(np.load(path +'/y.npy'), jax.devices('cpu')[0])
    
    ids = np.arange(0, len(x))
    ids = jax.random.permutation(rng, ids)
    
    x = x[ids]
    y = y[ids]
    x = process_data(x)
    
    y = jax.nn.one_hot(y, num_classes)

    split_point = int(test_size * len(x))

    x_test = x[0:split_point]
    y_test = y[0:split_point]

    x_train = x[split_point:]
    y_train = y[split_point:]


    return (x_train, y_train), (x_test, y_test)

def process_data(images):
    return images.mean(axis=-1)