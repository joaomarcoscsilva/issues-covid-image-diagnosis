import jax
from jax import numpy as np
import optax
from functools import partial
from tqdm import tqdm

def get_network_fns(net, optim, batch_size = 128, parallel = True, shape = (10, 256, 256, 3)):
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

    def loss_fn(params, state, x, y, training):
        """
        Applies the neural network to a batch, calculating the loss and accuracy.
        It also returns the updated network state, for example the batch norm statistics
        """

        # Applies the network to the batch and save the logits and the new network state (e.g. batch norm statistics)
        logits, state = net.apply(params, state, None, x, training)

        # Calculates metrics
        loss = optax.softmax_cross_entropy(logits = logits, labels = y).mean()
        acc = (logits.argmax(1) == y.argmax(1)).mean()

        return loss, (acc, state)
    
    

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    """ 
    Finds the gradient of the loss function on a batch. Also returns all the results in the network above.
    The returned values are in the format ((loss, (acc, state)), grad)
    """


    def update(params, state, optim_state, x, y):
        """
        Perform a neural network optimization step in a batch, returning the new paramaters and state.
        """

        # Calculate the gradient of the loss function in the batch, together with metrics and the new network state
        (loss, (acc, state)), grad = grad_fn(params, state, x, y, training = True)

        # If using parallelization, aggregates the gradient and state from all devices
        if parallel:
            state = jax.lax.pmean(state, 'parallel_dim')
            grad =  jax.lax.pmean(grad, 'parallel_dim')

        # Applies the updates to the parameters (possibly in multiple devices)
        updates, optim_state = optim.update(grad, optim_state, params)
        params = optax.apply_updates(params, updates)

        return params, state, optim_state, loss, acc

    
    # Applies either pmap or jit to the update function, depending on whether parallelization is used
    if parallel:
        update = jax.pmap(update, axis_name = 'parallel_dim', donate_argnums = [0,1,2])
    else:
        update = jax.pmap(update, donate_argnums = [0,1,2])

    def shard_array(array):
        """
        Split an array in pieces, placing each one in a different device.
        Used to send each TPU core a separate fraction of the batch
        """

        # Reshapes the array so that the first dimension is equal to the number of devices
        array = array.reshape(jax.local_device_count(), -1, *array.shape[1:])

        # Returns the sharded array
        return jax.device_put_sharded(list(array), jax.local_devices())


    def get_datagen(X, Y = None, include_last = False):
        """
        Creates a data generator that iterates through X (and possibly Y), returning batches in the correct devices.
        If include_last is true, includes the last batch even if its size is smaller than all the others.
        """

         # Finds the correct number of batches
        num_batches = len(X) // batch_size           
        if include_last:
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

    if parallel:
        p_apply_fn = jax.pmap(net.apply, static_broadcasted_argnums = (2,4,5))
        p_loss_fn = jax.pmap(loss_fn, static_broadcasted_argnums = 4)

    def predict(params, state, X, return_representation = False, training = True, verbose = True):
        """
        Applies the neural network in batches to every example in X.
        If return_representation is true, returns a latent embedding instead of the final layer. 
        """

        # Chooses the correct apply function depending on parallelization
        _apply_fn = p_apply_fn if parallel else net.apply
        
        # Gets a data generator for the dataset
        datagen, num_batches = get_datagen(X, include_last = True)
        
        # List with all the predictions
        preds = []

        # Applies the network to each batch
        for x in tqdm(datagen(), ncols = 120, total = num_batches, disable = not verbose):
            pred = _apply_fn(params, state, None, x, training, return_representation)[0]
            pred = pred.reshape(-1, *pred.shape[2:])
            preds.append(jax.device_put(pred, jax.devices('cpu')[0]))

        # Concatenates all predictions and places them on the cpu
        return jax.device_put(np.concatenate(preds), jax.devices('cpu')[0])


    def evaluate(params, state, X, Y, training = True, verbose = True):
        """
        Applies the neural network in batches to every example in X and Y, returing the average loss and accuracy.
        """
        
        # Chooses the correct loss function depending on parallelization
        _loss_fn = p_loss_fn if parallel else loss_fn
        
        # Gets a data generator for the dataset
        datagen, num_batches = get_datagen(X, Y, include_last = True)
        
        # Lists with all the metrics
        losses = []
        accs = []

        # Calculates the metrics for each batch
        for x, y in tqdm(datagen(), ncols = 120, total = num_batches, disable = not verbose):
            loss, (acc, _) = _loss_fn(params, state, x, y, training)
            losses.append(loss)
            accs.append(acc)
        
        # Returns the average results
        return np.array(losses).mean(), np.array(accs).mean()

    
    def train_epoch(params, state, optim_state, x_train, y_train, x_test = None, y_test = None, verbose = True):
        """
        Trains the neural network for an epoch.
        If x_test and y_test are passed, evaluates after training.
        """

        # Gets a data generator for the dataset
        datagen, num_batches = get_datagen(x_train, y_train, include_last = False)

        
        # Creates a progress bar for the epoch
        with tqdm(None, ncols = 120, total = num_batches, disable = not verbose) as bar:
            
            # Lists with all the metrics for training
            losses = []
            accs = []

            # Iterates the training set
            for x, y in datagen():
                
                # Performs the training step
                params, state, optim_state, _loss, _acc = update(params, state, optim_state, x, y)
                _loss, _acc = _loss.mean(), _acc.mean()
                
                # Updates the progress bar
                bar.update()
                bar.set_postfix({'loss': '%.2f' % _loss, 'acc': '%.2f' % _acc})

                # Saves the calculated metrics
                losses.append(_loss)
                accs.append(_acc)


            # If available, evaluates on the test set
            if x_test is not None and y_test is not None:
                loss, acc = evaluate(params, state, x_test, y_test, verbose = False)
                bar.set_postfix({'loss' : '%.2f' % np.array(losses).mean(), 'acc' : '%.2f' % np.array(accs).mean(), 'val_loss' : '%.2f' % loss, 'val_acc' : '%.2f' % acc})

        # Returns the new parameters and state
        return params, state, optim_state


    return init_fn, loss_fn, grad_fn, update, predict, evaluate, train_epoch 
                