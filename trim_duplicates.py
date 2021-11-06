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

# Given two matrices of shapes [n, dim] and [m, dim], returns a matrix [n, m] with all cosine similarities between the vectors

@partial(jax.vmap, in_axes = (None, 0))
@partial(jax.vmap, in_axes = (0, None))

def cosine_similarity(x,y):
    return jnp.dot(x,y) / jnp.sqrt(jnp.dot(x,x) * jnp.dot(y,y))

def compute_similarities(dataset, net_container, trained_model):
    print("Calculating embeddings...")

    # Finds the latent vectorsfor the entire dataset
    embeddings = net_container.predict(trained_model.params, trained_model.state, dataset.x_all, return_representation = True)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    print("Computing cosine similarities...")
    # Finds the cosine similarity between all images in the dataset
    sims = cosine_similarity(embeddings, embeddings)

    # Removes the diagonal of the matrix, since all images have a similarity of 1 with themselves
    sims = sims - np.eye(sims.shape[0])
    return sims

min_duplicates = 50
min_tol = 0.06
min_thresh = 0.95

def plot_similarities(dataset, sims, threshold=0.99):
    # Plots distribution of maximum similarity
    first_fig, axs = plt.subplots(1, 2, figsize = (15,5))
    
    max_sim = sims.max(0)
    sns.histplot(max_sim[max_sim > min_thresh], ax = axs[0]).set(title = 'Distribution of the maximum similarity for the images in the dataset')

    # Plots 'duplicates' according to threshold
    plt.title('Images with similarity greater than ' + str(threshold) + ' in each class')
    pd.Series(dataset.y_all[:sims.shape[0],][max_sim > threshold].argmax(1)).map(dict(zip(range(len(dataset.classnames)), dataset.classnames))).hist()
    first_fig.show()

    max_sims = sims.max(axis=1) - threshold
    max_sims_index = sims.argmax(axis=1)
    mask = (max_sims >= 0) & (max_sims <= 0.0005)
    indices = np.where(mask)[0]
    plots.compare_images(dataset.x_all[indices], dataset.x_all[max_sims_index[indices]], rows=10)

def remove_duplicates(dataset, sims):
    remove = set()

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

    plot_similarities(dataset, sims, threshold=thresh)
    plt.show()

    print("THRESH ALGORITHM OUTPUT", thresh)
    # THRESH ALGORITHM END

    # For each image not yet removed, add all of its duplicates to the removed set

    for i in tqdm(range(len(dataset.x_all))):
        if i not in remove:
            remove = remove.union(set(list(np.nonzero(sims[i] > thresh)[0])))

    print('trim_duplicates.remove_duplicates - Removed images:', len(remove), "({:.1f}%)".format(len(remove) / dataset.x_all.shape[0] * 100))

    # Finds the indices of the images that will not to be removed
    keep = set(range(len(dataset.x_all))).difference(remove)
    keep = np.array(list(keep))

    # These indices are for the x_all and y_all arrays.
    # The tensor x_all, for example, is the concatenation of x_test and x_train

    # Converts keep indices to be used in the train and test tensors
    keep_test = keep[keep < len(dataset.x_test)]
    keep_train = keep[keep >= len(dataset.x_test)] - len(dataset.x_test)

    # Removes the duplicates from the train and test sets
    x_train_curated = dataset.x_train[keep_train]
    y_train_curated = dataset.y_train[keep_train]

    x_test_curated = dataset.x_test[keep_test]
    y_test_curated = dataset.y_test[keep_test]

    return Dataset(x_train_curated, y_train_curated, x_test_curated, y_test_curated,
                    dataset.name + "_curated", dataset.classnames)
