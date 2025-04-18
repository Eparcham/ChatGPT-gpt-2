from datasets import load_dataset
from pyarrow.dataset import dataset

ds = load_dataset("roneneldan/TinyStories")
print(ds.cache_files)

train = ds['train']['text']
val = ds['validation']['text']

print(train[0])




