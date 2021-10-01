import numpy as np
import data
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
import utils

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

def plot_similarities(dataset, sims, threshold=0.99):
    # Plots distribution of maximum similarity
    first_fig, axs = plt.subplots(1, 2, figsize = (15,5))
    
    max_sim = sims.max(0)
    sns.histplot(max_sim, ax = axs[0]).set(title = 'Distribution of the maximum similarity for the images in the dataset')

    # Plots 'duplicates' according to threshold
    plt.title('Images with similarity greater than ' + str(threshold) + ' in each class')
    pd.Series(dataset.y_all[:sims.shape[0],][max_sim > threshold].argmax(1)).map(dict(zip(range(len(utils.CLASS_NAMES)), utils.CLASS_NAMES))).hist()
    first_fig.show()
    
    columns = 5
    rows = 5
    num_imgs_to_show = columns*rows

    # Gets distance of similarities to thresh
    thresh_dist = max_sim - threshold
    # Artificially increases dist of similarities smaller than threshold
    thresh_dist = np.array((thresh_dist < 0) * 10000 + thresh_dist)

    # Picks the N images with the smallest distance
    smallest_idx = np.argpartition(thresh_dist, num_imgs_to_show)

    # TODO: assert np.all(max_sim[smallest_idx,] > 0)
    second_fig = plt.figure(figsize=(16, 16))
    
    for j in range(0, int((columns*rows)/2)):
        img_idx = smallest_idx[j]
        img = dataset.x_all[img_idx,]
        second_fig.add_subplot(rows, columns, j*2+1)
        plt.imshow(img)

        idx_of_largest_sim = sims[img_idx,].argmax()
        similar_img = dataset.x_all[idx_of_largest_sim,]
        second_fig.add_subplot(rows, columns, j*2+2)
        plt.imshow(similar_img)
    
    plt.show()

def remove_duplicates(dataset, sims, threshold=0.99):
    remove = set()

    # For each image not yet removed, add all of its duplicates to the removed set

    for i in tqdm(range(len(dataset.x_all))):
        if i not in remove:
            remove = remove.union(set(list(np.nonzero(sims[i] > threshold)[0])))

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

    return Dataset(x_train_curated, y_train_curated, x_test_curated, y_test_curated)
