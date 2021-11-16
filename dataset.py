import jax
from jax import numpy as np
from glob import glob
import os

class Dataset:
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array
    x_all: np.array
    y_all: np.array

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_all = np.concatenate([self.x_train, self.x_test])
        self.y_all = np.concatenate([self.y_train, self.y_test])
    
    def five_fold(self, i):
        assert i >= 0 and i <= 4
        ds_size = self.x_all.shape[0]
        split_size = int(ds_size / 5)
        assert split_size > 5

        start = split_size*i
        end = split_size*(i+1)
        fold_x_test = self.x_all[start:end]
        fold_y_test = self.y_all[start:end]

        x_train_first_part = self.x_all[:start] if start > 0 else np.array([])
        x_train_second_part = self.x_all[end:] if end < ds_size else np.array([])

        y_train_first_part = self.y_all[:start] if start > 0 else np.array([])
        y_train_second_part = self.y_all[end:] if end < ds_size else np.array([])

        if start <= 0:
            fold_x_train = x_train_second_part
            fold_y_train = y_train_second_part
        elif end >= ds_size:
            fold_x_train = x_train_first_part
            fold_y_train = y_train_first_part
        else:
            fold_x_train = np.concatenate([x_train_first_part, x_train_second_part])
            fold_y_train = np.concatenate([y_train_first_part, y_train_second_part])
        
        assert fold_x_train.shape[0] <= split_size * 5 and fold_x_train.shape[0] >= split_size * 4
        assert fold_y_train.shape[0] <= split_size * 5 and fold_y_train.shape[0] >= split_size * 4
        assert fold_x_test.shape[0] == split_size and fold_y_test.shape[0] == split_size

        return Dataset(fold_x_train, fold_y_train, fold_x_test, fold_y_test)

    @staticmethod
    def load(dataset_name, rng, test_size = 0.2, num_classes = 4):
        """
        Loads the dataset to memory, already shuffled and split into train and test.
        """

        if 'x.npy' in os.listdir(dataset_name):
            x = jax.device_put(np.load(dataset_name + '/x.npy'), jax.devices('cpu')[0])
            y = jax.device_put(np.load(dataset_name +'/y.npy'), jax.devices('cpu')[0])
        
            ids = np.arange(0, len(x))
            ids = jax.random.permutation(rng, ids)
        
            x = x[ids]
            y = y[ids]
        
            y = jax.nn.one_hot(y, num_classes)

            split_point = int(test_size * len(x))

            x_test = x[0:split_point]
            y_test = y[0:split_point]

            x_train = x[split_point:]
            y_train = y[split_point:]
        
        else:
            x_train = jax.device_put(np.load(dataset_name + '/x_train.npy'), jax.devices('cpu')[0])
            y_train = jax.device_put(np.load(dataset_name +'/y_train.npy'), jax.devices('cpu')[0])
            x_test = jax.device_put(np.load(dataset_name + '/x_test.npy'), jax.devices('cpu')[0])
            y_test = jax.device_put(np.load(dataset_name +'/y_test.npy'), jax.devices('cpu')[0])

            y_train = jax.nn.one_hot(y_train, num_classes)
            y_test = jax.nn.one_hot(y_test, num_classes)

            ids_train = np.arange(0, len(x_train))
            ids_train = jax.random.permutation(rng, ids_train)

            x_train = x_train[ids_train]
            y_train = y_train[ids_train]

            ids_test = np.arange(0, len(x_test))
            ids_test = jax.random.permutation(rng, ids_test)

            x_test = x_test[ids_test]
            y_test = y_test[ids_test]

        return Dataset(x_train, y_train, x_test, y_test)


def shard_array(array):
    """
    Split an array in pieces, placing each one in a different device.
    Used to send each TPU core a separate fraction of the batch
    """
    # Reshapes the array so that the first dimension is equal to the number of devices
    array = array.reshape(jax.local_device_count(), -1, *array.shape[1:])
    # Returns the sharded array
    return jax.device_put_sharded(list(array), jax.local_devices())

def get_datagen(parallel, batch_size, X, Y = None, include_last = False):
    """
    Creates a data generator that iterates through X (and possibly Y), returning batches in the correct devices.
    If include_last is true, includes the last batch even if its size is smaller than all the others.
    """

    # Finds the correct number of batches
    num_batches = X.shape[0] // batch_size         
    if False: # ! TEMP: if include_last:
        num_batches += 1

    def datagen():
        """Data generator"""

        # Iterates through the dataset
        for i in range(num_batches):

            # Get a batch of X.
            # If parallelization is used, shard the batch between all devices. If not, place it in the first device.
            x = X[i * batch_size: (i+1) * batch_size]

            x = shard_array(x) if parallel else jax.device_put(x, jax.local_devices()[0])

            # If we are also iterating through Y, gets the corresponging batch
            if Y is not None:
                y = Y[i * batch_size: (i+1) * batch_size]
                y = shard_array(y) if parallel else jax.device_put(y, jax.local_devices()[0])
                # Returns the batch
                yield x, y
        
            else:
                # Returns the batch
                yield x
        
    return datagen, num_batches