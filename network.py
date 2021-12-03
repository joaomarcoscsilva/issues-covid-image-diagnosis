from typing import Callable, NamedTuple
from jax import numpy as np
import jax
import optax
from tqdm import tqdm
import dataset
import wandb
from wandb import Table
from sklearn.metrics import confusion_matrix
from IPython import embed
import plots
import pandas as pd
import seaborn as sns

class NetworkContainer(NamedTuple):
    init_fn: Callable
    loss_fn: Callable
    grad_fn: Callable
    update: Callable
    predict: Callable
    evaluate: Callable
    train_epoch: Callable


def create(net, optim, batch_size = 128, parallel = True, shape = (10, 256, 256, 3)):
    """
    Creates all the functions necessary to train the given neural network.
    """

    def init_fn(rng):
        """
        Initializes the starting parameters and state of a neural network and its optimizer.
        """    

        # Initializes parameters, network state and optimizer state
        params, state = net.init(rng, np.zeros(shape), is_training = True)
        optim_state = optim.init(params)

        # If using parallelization, create a copy of every variable in each device
        if parallel:
            params = jax.device_put_replicated(params, jax.local_devices())
            state = jax.device_put_replicated(state, jax.local_devices())
            optim_state = jax.device_put_replicated(optim_state, jax.local_devices())

        # If not, put all variables in the first device
        else:
            params = jax.device_put(params, jax.local_devices()[0])
            state = jax.device_put(state, jax.local_devices()[0])
            optim_state = jax.device_put(optim_state, jax.local_devices()[0])

        return params, state, optim_state

    def loss_fn(params, state, x, y, training, normalize = False):
        """
        Applies the neural network to a batch, calculating the loss and accuracy.
        It also returns the updated network state, for example the batch norm statistics
        """

        # Applies the network to the batch and save the logits and the new network state (e.g. batch norm statistics)
        logits, state = net.apply(params, state, None, x, training)

        # Calculates metrics
        loss = optax.softmax_cross_entropy(logits = logits, labels = y)
        acc = (logits.argmax(1) == y.argmax(1))

        non_normalized_loss = loss.mean()
        non_normalized_acc = acc.mean()
        
        y_probs = y.mean(0)
        weights = y_probs[y.argmax(1)]
        normalized_loss = loss / (weights * len(y_probs))
        normalized_acc = acc / (weights * len(y_probs))

        normalized_loss = normalized_loss.mean()
        normalized_acc = normalized_acc.mean()

        if normalize:
            loss = normalized_loss
        else:
            loss = non_normalized_loss

        metrics = np.array([non_normalized_loss, non_normalized_acc, normalized_loss, normalized_acc])
        

        return loss, (metrics, state, logits)
    
    

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    """ 
    Finds the gradient of the loss function on a batch. Also returns all the results in the network above.
    The returned values are in the format ((loss, (acc, state)), grad)
    """


    def update(params, state, optim_state, x, y, normalize = False):
        """
        Perform a neural network optimization step in a batch, returning the new paramaters and state.
        """

        # Calculate the gradient of the loss function in the batch, together with metrics and the new network state
        (loss, (metrics, state, pred)), grad = grad_fn(params, state, x, y, training = True, normalize = normalize)

        # If using parallelization, aggregates the gradient and state from all devices
        if parallel:
            state = jax.lax.pmean(state, 'parallel_dim')
            grad =  jax.lax.pmean(grad, 'parallel_dim')

        # Applies the updates to the parameters (possibly in multiple devices)
        updates, optim_state = optim.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return params, state, optim_state, loss, metrics

    
    # Applies either pmap or jit to the update function, depending on whether parallelization is used
    if parallel:
        update = jax.pmap(update, axis_name = 'parallel_dim', donate_argnums = [0,1,2], static_broadcasted_argnums = (5))
    else:
        # This is probably broken, as it should be jit instead of pmap. If using a single device, this needs to be fixed
        update = jax.pmap(update, donate_argnums = [0,1,2])

    if parallel:
        p_apply_fn = jax.pmap(net.apply, static_broadcasted_argnums = (2,4,5,6))
        p_loss_fn = jax.pmap(loss_fn, static_broadcasted_argnums = (4, 5))

    def predict(params, state, X, return_representation = False, return_gradcam = False, training = True, verbose = True):
        """
        Applies the neural network in batches to every example in X.
        If return_representation is true, returns a latent embedding instead of the final layer. 
        """

        # assert not (return_representation and training) For now, this assertion is disabled
        # assert not (return_gradcam and training) For now, this assertion is disabled

        # Chooses the correct apply function depending on parallelization
        _apply_fn = p_apply_fn if parallel else net.apply
        
        # Gets a data generator for the dataset
        datagen, num_batches = dataset.get_datagen(parallel, batch_size, X, include_last = True)
        
        # List with all the predictions
        preds = []

        # Applies the network to each batch
        for x in tqdm(datagen(), total = num_batches, disable = not verbose):
            pred = _apply_fn(params, state, None, x, training, return_representation, return_gradcam)[0]
            pred = pred.reshape(-1, *pred.shape[2:])
            preds.append(jax.device_put(pred, jax.devices('cpu')[0]))

        # Concatenates all predictions and places them on the cpu
        return jax.device_put(np.concatenate(preds), jax.devices('cpu')[0])


    def evaluate(params, state, X, Y, training = True, verbose = True, normalize = False):
        """
        Applies the neural network in batches to every example in X and Y, returing the average loss and accuracy.
        """

        # Chooses the correct loss function depending on parallelization
        _loss_fn = p_loss_fn if parallel else loss_fn
        
        # Gets a data generator for the dataset
        datagen, num_batches = dataset.get_datagen(parallel, batch_size, X, Y, include_last = True)
        
        # Lists with all the metrics
        losses = []
        metrics = []
        preds = []

        # Calculates the metrics for each batch
        for x, y in tqdm(datagen(), total = num_batches, disable = not verbose):
            _loss, (_metrics, _, pred) = _loss_fn(params, state, x, y, training, normalize)
            
            losses.append(_loss)
            metrics.append(_metrics)

            pred = pred.reshape(-1, *pred.shape[2:])
            preds.append(jax.device_put(pred, jax.devices('cpu')[0]))
        
        # Returns the average results

        losses = np.array(losses).mean()
        metrics = np.array(metrics).mean((0,1))

        return losses, metrics, jax.device_put(np.concatenate(preds), jax.devices('cpu')[0])

    METRICS = ['non_normalized_loss', 'non_normalized_acc', 'normalized_loss', 'normalized_acc']
    VAL_METRICS = ['val_' + s for s in METRICS]    
    TARGET_METRICS = ['target_' + s for s in METRICS]    

    def verify_optimizing_metric(optimizing_metric, metrics, val_metrics, target_metrics, current_metric):
        
        assert optimizing_metric[0] in '-+', "The optimizing metric must have a '+' or '-' as the first character indicating whether the metric must be maximized (+) or minimized (-)."

        sign = optimizing_metric[0]
        metric = optimizing_metric[1:]

        if metric in METRICS:
            val = metrics[METRICS.index(metric)]
        elif metric in VAL_METRICS:
            val = val_metrics[VAL_METRICS.index(metric)]
        elif metric in TARGET_METRICS:
            val = target_metrics[TARGET_METRICS.index(metric)]
        else:
            raise ValueError("The given optimizing metric is not supported.")

        if sign == '+':
            if current_metric is None or val > current_metric:
                return True, val
        else:
            if current_metric is None or val < current_metric:
                return True, val
        
        return False, current_metric
        

    def train_epoch(params, state, optim_state, x_train, y_train, x_test = None, y_test = None, 
    verbose = True, wandb_run = None, classnames = None, final_epoch = False,
    name = '', normalize = False, optimizing_metric = None, current_metric = None, x_target = None, y_target = None):
        """
        Trains the neural network for an epoch.
        If x_test and y_test are passed, evaluates after training.
        """

        def union_dict(*dicts):
            final = dict()
            for d in dicts:
                d = dict(d)
                for k in d:
                    final[k] = d[k]
            return final

        
        # Gets a data generator for the dataset
        datagen, num_batches = dataset.get_datagen(parallel, batch_size, x_train, y_train, include_last = False)

        final_acc = None
            
        # Creates a progress bar for the epoch
        with tqdm(None, ncols = 350, total = num_batches, disable = not verbose) as bar:
        
            # Lists with all the metrics for training
            losses = []
            metrics = []
            
            # Iterates the training set
            for x, y in datagen():
                
                # Performs the training step
                params, state, optim_state, _loss, _metrics = update(params, state, optim_state, x, y, normalize)
                _loss = _loss.mean()
                _metrics = _metrics.mean(0)

                # Updates the progress bar
                bar.update()
                bar.set_postfix(union_dict({'loss': ('%.2f' % _loss)}, 
                    zip(METRICS, map(lambda x: ('%.2f' % x),_metrics))))

                # Saves the calculated metrics
                losses.append(_loss)
                metrics.append(_metrics)
                
                if not wandb_run is None:
                    wandb_run.log(union_dict({'loss': float(_loss)}, zip(METRICS, map(float, _metrics))))

            # If available, evaluates on the test set
            if x_test is not None and y_test is not None:
                val_loss, val_metrics, logits = evaluate(params, state, x_test, y_test, verbose = False, normalize = normalize)
                
                bar.set_postfix(union_dict({'loss' : '%.2f' % np.array(losses).mean()},
                    zip(METRICS, map(lambda m: ('%.2f' % m), np.array(metrics).mean(0))),
                    zip(VAL_METRICS, map(lambda x: ('%.2f' % x), val_metrics))))

                conf_matrix = confusion_matrix(y_true = y_test[0:len(logits)].argmax(1), y_pred = logits.argmax(1), normalize = 'true')

                val_metrics_dict = dict(zip(VAL_METRICS, val_metrics))
                final_acc = val_metrics_dict['val_non_normalized_acc'] if not normalize else val_metrics_dict['val_normalized_acc']

                if not wandb_run is None:
                    wandb_run.log(union_dict({'val_loss': float(val_loss)},
                    zip(VAL_METRICS, map(float, val_metrics))))
            else:
                val_metrics = None

                # If available, evaluates on the target set
            if x_target is not None and y_target is not None and wandb_run is not None:
                target_loss, target_metrics, logits = evaluate(params, state, x_target, y_target, verbose = False, normalize = normalize)
                target_metrics_dict = dict(zip(TARGET_METRICS, target_metrics))
                wandb_run.log(target_metrics_dict)
            else:
                target_metrics = None

        if optimizing_metric is not None:
            improved, current_metric = verify_optimizing_metric(optimizing_metric, np.array(metrics).mean(0), val_metrics, target_metrics, current_metric)
            if improved:
                print(f'Reached maximum value of {optimizing_metric[1:]} so far.')
                return params, state, optim_state, True, current_metric
        
        if not final_acc is None and final_epoch:
            sns.heatmap(conf_matrix, annot = True, xticklabels = classnames, yticklabels = classnames)
            plots.wandb_log_img(wandb_run, "Confusion matrix")
                
        # Returns the new parameters and state
        return params, state, optim_state, False, current_metric


    return NetworkContainer(init_fn, loss_fn, grad_fn, update, predict, evaluate, train_epoch)
                
