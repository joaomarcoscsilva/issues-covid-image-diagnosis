from dataset import Dataset
import sys
import jax

r1, r2 = jax.random.split(jax.random.PRNGKey(1))
dataset_mendeley = Dataset.load("mendeley_curated", rng=r1, official_split=True, drop_classes = [1,3])
dataset_covidx = Dataset.load("covidx_curated", rng=r1, official_split=True, drop_classes = [1])

mixed = Dataset.merge([dataset_mendeley, dataset_covidx])
mixed.name = 'mendeley_covidx'
mixed.save('mendeley_covidx', split=True)

