import numpy as np
import dataset
import jax.numpy as jnp
import jax
from dataset import Dataset
import resnet
import haiku as hk
import optax
from functools import partial
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
import plots
from dataclasses import dataclass
import pickle

# Given two matrices of shapes [n, dim] and [m, dim], returns a matrix [n, m] with all cosine similarities between the vectors

@dataclass
class DuplicatesData:
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    rng: jax.random.PRNGKey
    indices: set

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        

@partial(jax.vmap, in_axes = (None, 0))
@partial(jax.vmap, in_axes = (0, None))

def cosine_similarity(x,y):
    return jnp.dot(x,y) / jnp.sqrt(jnp.dot(x,x) * jnp.dot(y,y))

def compute_similarities(dataset, net_container, trained_model, pixel_space):
    print("Calculating embeddings...")

    if not pixel_space:
        # Finds the latent vectors for the entire dataset
        embeddings = net_container.predict(trained_model.params, trained_model.state, dataset.x_all, return_representation = True)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings = dataset.x_all.reshape(dataset.x_all.shape[0], -1)

    print("Computing cosine similarities...")
    # Finds the cosine similarity between all images in the dataset
    sims = cosine_similarity(embeddings, embeddings)

    # Removes the diagonal of the matrix, since all images have a similarity of 1 with themselves
    sims = sims - np.eye(sims.shape[0])
    return sims

min_duplicates = 50
min_tol = 0.06
min_thresh = 0.95

def plot_similarities(dataset, sims, threshold, wandb_run):
    # Plots distribution of maximum similarity    
    plt.clf()
    max_sim = sims.max(0)
    ax = sns.histplot(max_sim[max_sim > min_thresh], bins=88)
    ax.set(title = 'Distribution of the maximum similarity for each images in the dataset')
    ax.set(xlim=(0.985, 1.0))
    ax.vlines(x=threshold, colors='purple', ls='dashed', ymin=0, ymax=ax.get_ylim()[1], label='duplicates threshold')
    plots.wandb_log_img(wandb_run, "Max histogram plot")

    # Plots 'duplicates' according to threshold
    plt.clf()
    plt.title('Images with similarity greater than ' + str(threshold) + ' in each class')
    pd.Series(dataset.y_all[:sims.shape[0],][max_sim > threshold].argmax(1)).map(dict(zip(range(len(dataset.classnames)), dataset.classnames))).hist()
    print(pd.Series(dataset.y_all[:sims.shape[0],][max_sim > threshold].argmax(1)).map(dict(zip(range(len(dataset.classnames)), dataset.classnames))).value_counts())
    plots.wandb_log_img(wandb_run, "Max similarity plot")

    max_sims = sims.max(axis=1) - threshold
    max_sims_index = sims.argmax(axis=1)
    mask = (max_sims >= 0) & (max_sims <= 0.0005)
    indices = np.where(mask)[0]
    plots.compare_images(dataset.x_all[indices], dataset.x_all[max_sims_index[indices]], rows=10)
    plots.wandb_log_img(wandb_run, "Images closest to threshold")

def remove_duplicates(suffix, dataset, sims, wandb_run=None):
    # THRESH ALGORITHM START
    max_sim = sims.max(0)
    step = 0.001
    bins = int((1-min_thresh)/step)
    hist, edges = jnp.histogram(max_sim[max_sim > min_thresh], bins=bins, density=False)
    total = hist.sum()
    hist_min = hist.min()
    acc = 0
    thresh = edges[-1]

    for i in reversed(range(hist.shape[0])):
        acc += hist[i]
        
        assert hist[i] > 0, "Is the step too small?"

        # 1 - Counts the number of duplicates that this thresh covers.
        # 2 - Ensures that hist[i] is very close to the minimum.
        # 3 - Ensures that hist[i] is less than hist[i-1]
        if acc >= min_duplicates and hist[i] <= hist_min + min_tol*total and hist[i-1] > hist[i]:
            break
        
        thresh = edges[i]

    plot_similarities(dataset, sims, threshold=thresh, wandb_run=wandb_run)
    # THRESH ALGORITHM END

    # For each image not yet removed, add all of its duplicates to the removed set
    
    one_duplicate = set()
    duplicate_groups = set()

    for i in tqdm(range(len(dataset.x_all))):
        if i not in one_duplicate:
            one_duplicate = one_duplicate.union(set(list(np.nonzero(sims[i] > thresh)[0])))
        duplicate_groups.add(tuple(sorted(list(np.nonzero(sims[i] > thresh)[0]) + [i])))
    
    dup_count = len(dataset.x_all) - len(duplicate_groups)
    
    def tests_in_set(s):
        amount = 0
        for i in s:
            if i < len(dataset.x_test):
                amount += 1
        return amount

    def trains_in_set(s):
        amount = 0
        for i in s:
            if i >= len(dataset.x_test):
                amount += 1
        return amount

    def both_in_set(s):
        amount = 0
        train = False
        test = False
        for i in s:
            if i < len(dataset.x_test):
                test = True
                amount += 1
            else:
                train = True
        return amount if (train and test) else 0

    num_test_duplicates = sum([tests_in_set(g) for g in duplicate_groups if len(g) > 1])
    num_train_duplicates = sum([trains_in_set(g) for g in duplicate_groups if len(g) > 1])
    num_leaked_duplicates = sum([both_in_set(g) for g in duplicate_groups if len(g) > 1])

    print('Unique images:', len(duplicate_groups))
    print('Removed images:', dup_count)
    print('Duplicates found in test set:', num_test_duplicates)
    print('Duplicates found in train set:', num_train_duplicates)
    print('Leaked examples from the test set:', num_leaked_duplicates)

    if wandb_run is not None:
        dup_data = DuplicatesData(dataset.rng, duplicate_groups)
        fname = "dup_data/" + wandb_run.name + suffix + ".pickle"
        dup_data.save(fname)
        wandb_run.save(fname)

        wandb_run.log({
            'unique_images' + suffix: len(duplicate_groups),
            'removed_images' + suffix: dup_count,
            'test_duplicates' + suffix: num_test_duplicates,
            'train_duplicates:' + suffix: num_train_duplicates,
            'leaked_duplicates' + suffix: num_leaked_duplicates
        })

    assert np.all(dataset.x_all[0:len(dataset.x_test)] == dataset.x_test), "Something is very very very wrong."

    keep = [i for i in range(len(dataset.x_all)) if i not in one_duplicate]
    keep_test =  jnp.asarray([i for i in keep if i < len(dataset.x_test)], dtype=jnp.int32)
    keep_train = jnp.asarray([i for i in keep if i >= len(dataset.x_test)], dtype=jnp.int32)
    
    # Removes the duplicates from the train and test sets
    x_train_curated = dataset.x_all[keep_train]
    y_train_curated = dataset.y_all[keep_train]
    paths_train_curated = dataset.paths_all[keep_train]

    x_test_curated = dataset.x_all[keep_test]
    y_test_curated = dataset.y_all[keep_test]
    paths_test_curated = dataset.paths_all[keep_test]

    return Dataset(x_train_curated, y_train_curated, x_test_curated, y_test_curated,
                    dataset.name + "_curated", dataset.classnames, dataset.rng, paths_train_curated, paths_test_curated), duplicate_groups