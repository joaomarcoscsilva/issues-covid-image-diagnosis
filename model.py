from typing import Mapping, NamedTuple
import numpy as np
import jax.numpy as jnp
import jax
import os
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import utils
import matplotlib.pyplot as plt
import resnet
import haiku as hk
import optax

class ModelContainer(NamedTuple):
    name: str
    params: Mapping
    state: Mapping
    optim_state: Mapping
    x_train_proc: np.array
    x_test_proc: np.array
    y_train: np.array
    y_test: np.array

def init_net_and_optim(x_train, num_classes, batch_size):
    def forward(batch, is_training, return_representation = False, return_gradcam = False, gradcam_counterfactual = False):
        net = resnet.ResNet18(num_classes = num_classes, resnet_v2 = True)
        if return_representation:
            return net.embedding(batch, is_training, embedding_depth=0)
        elif return_gradcam:
            return net.gradcam(batch, is_training, gradcam_depth=0, counterfactual=gradcam_counterfactual)
        else:
            return net(batch, is_training)
    
    net = hk.transform_with_state(forward)
    schedule = optax.cosine_decay_schedule(1e-1, 30 * (len(x_train) // batch_size))
    optim = optax.adamw(schedule, weight_decay = 1e-3)

    return net, optim

def get_persistent_fields(model):
    return (model.name, model.params, model.state, model.optim_state)

def train_model(name, net_container, process_fn, dataset, num_epochs = 30, rng = jax.random.PRNGKey(42)) -> ModelContainer:
    """Trains the network specified at net_container, in the given dataset.
       If models/name exists, returns the cached version. Otherwise, trains the model then saves it to model/name.

    Returns:
        ModelContainer: The trained model.
    """

    x_train_proc = process_fn(dataset.x_train)
    x_test_proc = process_fn(dataset.x_test)

    dst_path = "models/" + name + ".pickle"
    if os.path.exists(dst_path):
        with open(dst_path, "rb") as f:
            print("Model loaded from", dst_path)
            loaded_model = pickle.load(f)
            return ModelContainer(*loaded_model, x_train_proc, x_test_proc, dataset.y_train, dataset.y_test)

    params, state, optim_state = net_container.init_fn(jax.random.split(rng)[0])

    # Train the model for N epochs on the dataset
    for _ in range(num_epochs):
        params, state, optim_state = net_container.train_epoch(params, state, optim_state, x_train_proc,
                                                               dataset.y_train, x_test_proc, dataset.y_test)
    
    model = ModelContainer(name, params, state, optim_state, x_train_proc, x_test_proc, dataset.y_train, dataset.y_test)
    
    with open(dst_path, "wb") as f:
        print("Model saved to", dst_path)
        pickle.dump(get_persistent_fields(model), f)

    return model

def plot_confusion_matrix(model_container, y_pred):
    sns.heatmap(confusion_matrix(model_container.y_test.argmax(1), y_pred.argmax(1), normalize = 'true'),
                                 annot = True, xticklabels = utils.CLASS_NAMES, yticklabels = utils.CLASS_NAMES)
    plt.show()