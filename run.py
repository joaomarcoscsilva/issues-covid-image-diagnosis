import matplotlib.pyplot as plt
import argparse
import json
import hashlib
import sys
import os

# Use argparse to get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('config_file', type=str, help = "The path to a json configuration file.")
parser.add_argument('--wandb', action = 'store_true', help = "Whether or not to log the experiment to weights and biases.", default = False)
parser.add_argument('--save', help = "Whether or not to save the trained weights.", action = 'store_true', default = False)
parser.add_argument('--load', help = "Path of a pickle file containing pretrained weights.", default = False)
parser.add_argument('--name', help = "Name of the run. If undefined, will create a random hash of the configuration.", default = "")
parser.add_argument('-f', '--force', help = "If this is set, any existing weights with the same name will be removed", action = 'store_true', default = False)
parser.add_argument('--cv', help = "The number of cross-validation folds to use. If unset, doesn't use cross-validation.", type = int, default = None)
parser.add_argument('--cv-id', help = argparse.SUPPRESS, type = int, default = None)

parser.add_argument('--dedup', help = "If set, will run the dataset deduplication procedure.", action = 'store_true', default = False)
parser.add_argument('--save-dedup', help = "If set, will save the deduplicated dataset to the specified directory.", default = None)
parser.add_argument('--split-dedup', help = "If set, the deduplicated dataset will be split into train and test sets.", action = 'store_true', default = False)
parser.add_argument('--pixel-space', help = "If set, will run the deduplication in pixel space. This option also skips training as a whole.", action = 'store_true', default = False)

args = parser.parse_args()

assert not (args.cv is None and args.cv_id is not None), "You must specify the amount of cross-validation folds using the --cv command line option."
assert not ((args.save_dedup or args.pixel_space) and not args.dedup), "To run a deduplication procedure, pass the --dedup command line option."
assert not (args.split_dedup and not args.save_dedup), "To split a deduplicated dataset, pass the --save-dedup command line option as well."

# If args.cv is not None but args.cv_id is None, then we want the script to call itself multiple times to perform cross-validation
if args.cv is not None and args.cv_id is None:
    for i in range(args.cv):
        call_argv = ['python3'] + sys.argv + ['--cv-id %d' % i]
        os.system(' '.join(call_argv))
    exit()
        

# Loads the configuration file
with open(args.config_file, 'r') as f:
    config = json.load(f)



name = args.name if args.name != "" else hashlib.shake_128(json.dumps(config, sort_keys = True).encode()).hexdigest(5)
config['name'] = name

import wandb
import jax
assert jax.local_device_count() == 8, "No TPU available"

from dataset import Dataset
import model, network
import trim_duplicates

# Initialize the random seed
r1, r2 = jax.random.split(jax.random.PRNGKey(1))

data = Dataset.load("data/" + config['dataset'], rng=r1, drop_classes = config['drop_classes'], official_split = config['official_split'])
if config['target_dataset'] is not None:
    target_data = Dataset.load("data/" + config['target_dataset'], rng=r2, drop_classes = config['target_drop_classes'], official_split = config['target_official_split'])
else:
    target_data = None


if args.cv_id is not None:
    config['cv_id'] = args.cv_id
    config['dataset'] = config['dataset'] + '_' + str(args.cv_id)
    config['group'] = config['name']
    config['name'] = config['name'] + '_' + str(args.cv_id)

wandb_run = wandb.init(project='xrays', entity='usp-covid-xrays', reinit=True, config = config) if args.wandb else None

if not args.pixel_space:

    print('Doing run', config['name'])

    rng = jax.random.PRNGKey(config['random_seed'] + (args.cv_id if args.cv_id is not None else 0))

    net, optim = model.init_net_and_optim(data, config['batch_size'], initial_lr = config['initial_lr'], num_epochs = config['num_epochs'])
    net_container = network.create(net, optim, config['batch_size'], shape = (10, config['resolution'], config['resolution'], 3))

    

    trained_model = model.train_model(config['name'] if args.save else '', net_container, lambda x: x, data, masks = None, classnames = data.classnames, num_epochs = config['num_epochs'],
                                    wandb_run = wandb_run, rng = rng,
                                    normalize = config['normalize_loss'], optimizing_metric = config['optimizing_metric'], validation_size = config['validation_size'],
                                    target_data = target_data, force_save = args.force)

    if config['validation_size'] is not None and config['validation_size'] > 0:
        model.evaluate_model(net_container, trained_model, data.x_test, data.y_test, data.classnames, prefix = 'test', wandb_run = wandb_run)
        plt.show()

    if config['target_dataset'] is not None:
        model.evaluate_model(net_container, trained_model, target_data.x_all, target_data.y_all, target_data.classnames, prefix = 'target', wandb_run = wandb_run)
        plt.show()

if args.dedup:
    if args.pixel_space:
        sims = trim_duplicates.compute_similarities(data, None, None, pixel_space = True)
    else:
        sims = trim_duplicates.compute_similarities(data, net_container, trained_model, pixel_space = False)
    curated_data, duplicate_groups = trim_duplicates.remove_duplicates('', data, sims, wandb_run)

    if args.save_dedup is not None:
        curated_data.save(args.save_dedup, args.split_dedup)
        

if wandb_run is not None:
    wandb_run.finish()