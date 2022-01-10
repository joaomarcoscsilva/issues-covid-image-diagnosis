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
from dataset import train_test_split
from copy import deepcopy
import plots
from flax2haiku import load_imagenet_resnet

class ModelContainer(NamedTuple):
    name: str
    params: Mapping
    state: Mapping
    optim_state: Mapping

def init_net_and_optim(data, batch_size, initial_lr = 1e-1, num_epochs = 30):
    num_classes = len(data.classnames)

    def forward(batch, is_training, return_representation = False, return_gradcam = False, gradcam_counterfactual = False):
        net = resnet.ResNet18(num_classes = num_classes, resnet_v2 = False)
        if return_representation:
            return net.embedding(batch, is_training, embedding_depth=0)
        elif return_gradcam:
            return net.gradcam(batch, is_training, gradcam_depth=0, counterfactual=gradcam_counterfactual)
        else:
            return net(batch, is_training)
    
    net = hk.transform_with_state(forward)
    schedule = optax.cosine_decay_schedule(initial_lr, num_epochs * (len(data.x_train) // batch_size))
    optim = optax.adamw(schedule, weight_decay = 1e-3)

    return net, optim

def get_persistent_fields(model):
    return (model.name, model.params, model.state, model.optim_state)

def train_model(name, net_container, process_fn, dataset, num_epochs = 30, rng = jax.random.PRNGKey(42), masks = None, wandb_run = None, classnames = None,
                normalize = False, optimizing_metric = None, validation_size = None, target_datas = [], force_save = False,
                initialization = None, save_weights_to_wandb = False) -> ModelContainer:
    """Trains the network specified at net_container, in the given dataset.
       If models/name exists, returns the cached version. Otherwise, trains the model then saves it to model/name.

    Returns:
        ModelContainer: The trained model.
    """

    x_train_proc = process_fn(dataset.x_train)
    x_test_proc = process_fn(dataset.x_test)

    if name != '':
        dst_path = "models/" + name + ".pickle"
        if os.path.exists(dst_path) and not force_save:
            with open(dst_path, "rb") as f:
                print("Model loaded from", dst_path)
                loaded_model = pickle.load(f)
                return loaded_model

    if initialization is None:
        params, state, optim_state = net_container.init_fn(jax.random.split(rng)[0])

    else:
        params, state, optim_state = net_container.init_fn(jax.random.split(rng)[0])

        if initialization == 'imagenet':
            old_params = params
            params = load_imagenet_resnet(params)
        else:
            with open(initialization, "rb") as f:
                params, state = get_persistent_fields(pickle.load(f))[1:3]

    rng = jax.random.split(rng)[0]

    current_metric = None

    # Train the model for N epochs on the dataset
    for epoch_i in range(num_epochs):
        
        if masks is not None:
            _masks = masks[jax.random.choice(rng, len(masks), (len(x_train_proc),))]
            rng = jax.random.split(rng)[0]
            _x_train = x_train_proc * _masks
        else:
            _x_train = x_train_proc

        if validation_size is not None:
            _x_train, _x_test, _y_train, _y_test = train_test_split(_x_train, dataset.y_train, test_size = validation_size, rng = rng)
            rng = jax.random.split(rng)[0]
        else:
            _y_train = dataset.y_train
            _x_test = x_test_proc
            _y_test = dataset.y_test

        _x_targets = [process_fn(target_data.x_all) for target_data in target_datas]
        _y_targets = [target_data.y_all for target_data in target_datas]
        _target_names = [target_data.name for target_data in target_datas]

        params, state, optim_state, best_epoch, current_metric = net_container.train_epoch(params, state, optim_state, _x_train,
                                                               _y_train, _x_test, _y_test,
                                                               wandb_run = wandb_run, classnames = dataset.classnames,
                                                               name = name, normalize = normalize, 
                                                               optimizing_metric = optimizing_metric, current_metric = current_metric,
                                                               final_epoch = epoch_i == num_epochs-1, current_epoch = epoch_i,
                                                               x_targets = _x_targets, y_targets = _y_targets, target_names = _target_names)
                                                               
        
        if optimizing_metric is None or best_epoch:
            model_pk = pickle.dumps(ModelContainer(name, params, state, optim_state))
            
    if name != '':
        with open(dst_path, "wb") as f:
            print("Model saved to", dst_path)
            f.write(model_pk)
        if save_weights_to_wandb and wandb_run is not None:
            wandb.save(dst_path)
    
    return pickle.loads(model_pk)

def plot_confusion_matrix(y_test, y_pred, classnames):
    sns.heatmap(confusion_matrix(y_test.argmax(1), y_pred.argmax(1), normalize = 'true'),
                                 annot = True, xticklabels = classnames, yticklabels = classnames)
    plt.show()

def evaluate_model(net_container, model_container, x, y, classnames, prefix = '', wandb_run = None):
    METRICS = ['non_normalized_loss', 'non_normalized_acc', 'normalized_loss', 'normalized_acc']
    if prefix != '':
        METRICS = [prefix + '_' + metric for metric in METRICS]
        confusion_name = prefix.capitalize() + ' Confusion Matrix'

    metrics, logits = net_container.evaluate(model_container.params, model_container.state, x, y, verbose = False, normalize = False)[1:]
    conf_matrix = confusion_matrix(y_true = y[0:len(logits)].argmax(1), y_pred = logits.argmax(1), normalize = 'true')
    metrics_dict = dict(zip(METRICS, metrics))

    print(metrics_dict)
    
    plt.clf()
    sns.heatmap(conf_matrix, annot = True, xticklabels = classnames, yticklabels = classnames).set(title = confusion_name)

    if wandb_run is not None:
        wandb_run.log(metrics_dict)
        plots.wandb_log_img(wandb_run, confusion_name)

    plt.show()