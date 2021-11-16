from pydoc import classname
from typing import Mapping, NamedTuple
import numpy as np
import jax.numpy as jnp
import jax
import os
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import resnet
import haiku as hk
import optax
from tqdm import tqdm
import wandb

class ModelContainer(NamedTuple):
    name: str
    params: Mapping
    state: Mapping
    optim_state: Mapping
    x_train_proc: np.array
    x_test_proc: np.array
    y_train: np.array
    y_test: np.array

def init_net_and_optim(x_train, num_classes, batch_size, initial_lr = 1e-1):
    def forward(batch, is_training, return_representation = False, return_gradcam = False, gradcam_counterfactual = False):
        net = resnet.ResNet18(num_classes = num_classes, resnet_v2 = True)
        if return_representation:
            return net.embedding(batch, is_training, embedding_depth=0)
        elif return_gradcam:
            return net.gradcam(batch, is_training, gradcam_depth=0, counterfactual=gradcam_counterfactual)
        else:
            return net(batch, is_training)
    
    net = hk.transform_with_state(forward)
    schedule = optax.cosine_decay_schedule(initial_lr, 30 * (len(x_train) // batch_size))
    optim = optax.adamw(schedule, weight_decay = 1e-3)

    return net, optim

def get_persistent_fields(model):
    return (model.name, model.params, model.state, model.optim_state)

def train_model(name, net_container, process_fn, dataset, num_epochs = 30, rng = jax.random.PRNGKey(42), masks = None, wandb_run = None, class_names = utils.CLASS_NAMES, normalize = False, optimizing_metric = None) -> ModelContainer:
    """Trains the network specified at net_container, in the given dataset.
       If models/name exists, returns the cached version. Otherwise, trains the model then saves it to model/name.

    Returns:
        ModelContainer: The trained model.
    """

    x_train_proc = process_fn(dataset.x_train)
    x_test_proc = process_fn(dataset.x_test)

    if name != '':
        dst_path = "models/" + name + ".pickle"
        if os.path.exists(dst_path):
            with open(dst_path, "rb") as f:
                print("Model loaded from", dst_path)
                loaded_model = pickle.load(f)
                return ModelContainer(*loaded_model, x_train_proc, x_test_proc, dataset.y_train, dataset.y_test)

    params, state, optim_state = net_container.init_fn(jax.random.split(rng)[0])
    
    current_metric = None

    # Train the model for N epochs on the dataset
    for epoch_i in range(num_epochs):
        
        if masks is not None:
            _masks = masks[jax.random.choice(rng, len(masks), (len(x_train_proc),))]
            rng = jax.random.split(rng)[0]
            _x_train = x_train_proc * _masks
        else:
            _x_train = x_train_proc
    
        params, state, optim_state, best_epoch, current_metric = net_container.train_epoch(params, state, optim_state, _x_train,
                                                               dataset.y_train, x_test_proc, dataset.y_test,
                                                               wandb_run = wandb_run, classnames = dataset.classnames,
                                                               name = name, normalize = normalize, 
                                                               optimizing_metric = optimizing_metric, current_metric = current_metric,
                                                               final_epoch=epoch_i == num_epochs-1)

        if optimizing_metric is None or best_epoch: 
            model = ModelContainer(name, params, state, optim_state, x_train_proc, x_test_proc, dataset.y_train, dataset.y_test)
            
    if name != '':
        with open(dst_path, "wb") as f:
            print("Model saved to", dst_path)
            pickle.dump(get_persistent_fields(model), f)
        if log_wandb:
            wandb.save(dst_path)        
    
    return model

def plot_confusion_matrix(model_container, y_pred, classnames):
    sns.heatmap(confusion_matrix(model_container.y_test.argmax(1), y_pred.argmax(1), normalize = 'true'),
                                 annot = True, xticklabels = classnames, yticklabels = classnames)
    plt.show()
