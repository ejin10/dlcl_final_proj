import torch
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, DatasetDict, load_from_disk
import html2text
from transformers import BertTokenizerFast
import pickle as pkl
import os

# CONSTANTS
TRAIN_SPLIT = 0.7
SMALL = False

def html2txt(article):
    article['text'] = h.handle(article['text'])
    return article

def combine(article):
    article['text'] = article['title'].strip() + '. ' + article['text'].strip()
    return article

def limit(article):
    article['input_ids'] = tok.encode(article['text'], max_length=512, padding='max_length', truncation=True)
    return article

# Load dataset
if SMALL:
  dataset = load_dataset('hyperpartisan_news_detection', 'bypublisher', split='train[:10000]')
else:
  dataset = load_dataset('hyperpartisan_news_detection', 'bypublisher', split='train+validation')

print(f'num samples: {len(dataset)}')

# Train/val/test split
num_samples = len(dataset)
num_train = int(num_samples * TRAIN_SPLIT)
num_val = int(num_samples * (1 - TRAIN_SPLIT) / 2)

ds = DatasetDict(
    train=dataset.shuffle(seed=1111).select(range(num_train)),
    val=dataset.shuffle(seed=1111).select(range(num_train, num_train + num_val)),
    test=dataset.shuffle(seed=1111).select(range(num_train + num_val, num_samples))
)

for split in ds.keys():
  print(f'num {split}: {len(ds[split])}')


# Remove HTML tags from article text
h = html2text.HTML2Text()
h.ignore_links = True
ds = ds.map(html2txt, writer_batch_size=100)

# Combine article title and text
ds = ds.map(combine, writer_batch_size=100)

# Limit article text to 512 BERT tokens
tok = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
ds = ds.map(limit, writer_batch_size=100)

# Remove unused columns
ds = ds.remove_columns(['title', 'url', 'published_at', 'hyperpartisan'])
ds = ds.rename_column('bias', 'label')

# Save full dataset
CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, 'data')

if not os.path.isdir(DATA_DIR):
  os.mkdir(DATA_DIR)

if SMALL:
  save_to = os.path.join(DATA_DIR, 'small_dataset_ids')
else:
  save_to = os.path.join(DATA_DIR, 'full_dataset_ids')

ds.save_to_disk(save_to) 
print(f'saved to {save_to}')