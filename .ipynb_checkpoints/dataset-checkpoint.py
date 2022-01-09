import pickle
import jax
from jax import numpy as jnp
import numpy as np
from glob import glob
import os
from IPython import embed

class Dataset:
    x_train: jnp.array
    y_train: jnp.array
    x_test: jnp.array
    y_test: jnp.array
    x_all: jnp.array
    y_all: jnp.array
    paths_train: np.array
    paths_test: np.array
    paths_all: np.array
    name: str
    classnames: list
    rng: jax.random.PRNGKey

    def __init__(self, x_train, y_train, x_test, y_test, name, classnames, rng, paths_train, paths_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_all = jnp.concatenate([self.x_test, self.x_train])
        self.y_all = jnp.concatenate([self.y_test, self.y_train])
        self.name = name
        self.classnames = classnames
        self.rng = rng
        self.paths_train = paths_train
        self.paths_test = paths_test
        self.paths_all = np.concatenate([self.paths_test, self.paths_train])
    
    def k_fold(self, i, num_folds = 5):
        fold_size = self.x_all.shape[0] // num_folds
        fold_x_train = jnp.concatenate([self.x_all[:i*fold_size], self.x_all[(i+1)*fold_size:]])
        fold_y_train = jnp.concatenate([self.y_all[:i*fold_size], self.y_all[(i+1)*fold_size:]])
        fold_x_test = self.x_all[i*fold_size:(i+1)*fold_size]
        fold_y_test = self.y_all[i*fold_size:(i+1)*fold_size]
        fold_paths_train = np.concatenate([self.paths_all[:i*fold_size], self.paths_all[(i+1)*fold_size:]])
        fold_paths_test = self.paths_all[i*fold_size:(i+1)*fold_size]
        return Dataset(fold_x_train, fold_y_train, fold_x_test, fold_y_test, self.name + '_' + str(i+1), self.classnames, self.rng, fold_paths_train, fold_paths_test)
    
    def five_fold(self, i):
        return self.k_fold(i, num_folds = 5)

    @staticmethod
    def load(dataset_name, rng, test_size = 0.2, drop_classes = [], official_split = True):
        """
        Loads the dataset to memory, already shuffled and split into train and test.
        """
        
        with open(dataset_name + '/metadata.pickle', 'rb') as f:
            metadata = pickle.load(f)

        num_classes = len(metadata['classnames'])
        metadata['classnames'] = [metadata['classnames'][i] for i in range(len(metadata['classnames'])) if i not in drop_classes]
        drop_classes = jnp.array(drop_classes)
        keep_classes = jnp.array([i for i in range(num_classes) if i not in drop_classes])
        
        if 'x.npy' in os.listdir(dataset_name) or not official_split:
            
            if official_split:
                'x_train.npy' in os.listdir(dataset_name) and 'x_test.npy' in os.listdir(dataset_name), 'Setting official_split to False is only supported for datasets with an official train-test split'
                x_train = jax.device_put(jnp.load(dataset_name + '/x_train.npy'), jax.devices('cpu')[0])
                y_train = jax.device_put(jnp.load(dataset_name +'/y_train.npy'), jax.devices('cpu')[0])
                x_test = jax.device_put(jnp.load(dataset_name + '/x_test.npy'), jax.devices('cpu')[0])
                y_test = jax.device_put(jnp.load(dataset_name +'/y_test.npy'), jax.devices('cpu')[0])

                x = jnp.concatenate([x_train, x_test])
                y = jnp.concatenate([y_train, y_test])
                paths = np.concatenate([metadata['paths_train'], metadata['paths_test']])

            else:
                x = jax.device_put(jnp.load(dataset_name + '/x.npy'), jax.devices('cpu')[0])
                y = jax.device_put(jnp.load(dataset_name +'/y.npy'), jax.devices('cpu')[0])
                paths = metadata['paths']

            
            y = jax.nn.one_hot(y, num_classes)
            
            paths = paths[jnp.isin(y.argmax(1), keep_classes)]
            x = x[jnp.isin(y.argmax(1), keep_classes)]
            y = y[jnp.isin(y.argmax(1), keep_classes)]
            
            
            

            x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(x, y, paths, test_size = test_size, rng = rng)

            y_train = y_train[:, keep_classes]
            y_test = y_test[:, keep_classes]

        else:
            x_train = jax.device_put(jnp.load(dataset_name + '/x_train.npy'), jax.devices('cpu')[0])
            y_train = jax.device_put(jnp.load(dataset_name +'/y_train.npy'), jax.devices('cpu')[0])
            x_test = jax.device_put(jnp.load(dataset_name + '/x_test.npy'), jax.devices('cpu')[0])
            y_test = jax.device_put(jnp.load(dataset_name +'/y_test.npy'), jax.devices('cpu')[0])
            paths_train = metadata['paths_train']
            paths_test = metadata['paths_test']

            y_train = jax.nn.one_hot(y_train, num_classes)
            y_test = jax.nn.one_hot(y_test, num_classes)

            ids_train = jnp.arange(0, len(x_train))
            ids_train = ids_train[jnp.isin(y_train.argmax(1), keep_classes)]
            ids_train = jax.random.permutation(rng, ids_train)

            x_train = x_train[ids_train]
            y_train = y_train[ids_train]
            paths_train = paths_train[ids_train]

            ids_test = jnp.arange(0, len(x_test))
            ids_test = ids_test[jnp.isin(y_test.argmax(1), keep_classes)]
            ids_test = jax.random.permutation(rng, ids_test)

            x_test = x_test[ids_test]
            y_test = y_test[ids_test]
            paths_test = paths_test[ids_test]

            y_train = y_train[:, keep_classes]
            y_test = y_test[:, keep_classes]
        
        return Dataset(x_train, y_train, x_test, y_test, dataset_name, metadata['classnames'], rng, paths_train, paths_test)

    def save(self, dir, split = False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        if not split:
            jnp.save(dir + '/x.npy', self.x_all)
            jnp.save(dir + '/y.npy', self.y_all.argmax(1))
            metadata = {'classnames': self.classnames, 'paths': self.paths_all}
            with open(dir + '/metadata.pickle', 'wb') as f:
                pickle.dump(metadata, f)
        
        else:
            jnp.save(dir + '/x_train.npy', self.x_train)
            jnp.save(dir + '/y_train.npy', self.y_train.argmax(1))
            jnp.save(dir + '/x_test.npy', self.x_test)
            jnp.save(dir + '/y_test.npy', self.y_test.argmax(1))
            metadata = {'classnames': self.classnames, 'paths_train': self.paths_train, 'paths_test': self.paths_test}
            with open(dir + '/metadata.pickle', 'wb') as f:
                pickle.dump(metadata, f)

    def get_finetuning_dataset(self, train_samples = None, test_samples = None):
        
        assert (train_samples is None) ^ (test_samples is None), 'Either train_samples or test_samples must be None'

        if test_samples is None:
            test_samples = self.x_all.shape[0] - train_samples

        x_test = self.x_all[:test_samples]
        y_test = self.y_all[:test_samples]
        paths_test = self.paths_all[:test_samples]
        
        x_train = self.x_all[test_samples:]
        y_train = self.y_all[test_samples:]
        paths_train = self.paths_all[test_samples:]

        return Dataset(x_train, y_train, x_test, y_test, self.dataset_name, self.classnames, self.rng, paths_train, paths_test)
        
        

def train_test_split(*args, test_size = 0.2, rng = None):
    ids = np.arange(0, len(args[0]))
    if rng is not None:
        ids = jax.random.permutation(rng, ids)

    split_point = int(test_size * len(args[0]))

    results = []

    for arg in args:
        results.append(arg[ids[split_point:]])
        results.append(arg[ids[:split_point]])

    return results
    

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
    if include_last: 
        num_batches += 1

    def datagen():
        """Data generator"""

        # Iterates through the dataset
        for i in range(num_batches):

            # Get a batch of X.
            x = X[i * batch_size: (i+1) * batch_size]

            # Pads the batch if necessary
            if x.shape[0] != batch_size:
                x = jnp.concatenate([x, jnp.zeros((batch_size - x.shape[0], *x.shape[1:]))])

            # If parallelization is used, shard the batch between all devices. If not, place it in the first device.
            x = shard_array(x) if parallel else jax.device_put(x, jax.local_devices()[0])

            # If we are also iterating through Y, gets the corresponging batch
            if Y is not None:
                y = Y[i * batch_size: (i+1) * batch_size]

                # Pads the batch if necessary
                if y.shape[0] != batch_size:
                    y = jnp.concatenate([y, jnp.zeros((batch_size - y.shape[0], *y.shape[1:]))])

                y = shard_array(y) if parallel else jax.device_put(y, jax.local_devices()[0])
                # Returns the batch
                yield x, y
        
            else:
                # Returns the batch
                yield x
        
    return datagen, num_batches