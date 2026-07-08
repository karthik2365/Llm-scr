import os
import urllib.request

if not os.path.exists("verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    file_path = "verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("verdict.txt", "r") as f:
    raw_text = f.read()

import re

text = "hello world this is a test"

token_pattern = r'([,.:;?_!"()\']|--|\s)'

result = re.split(token_pattern, raw_text)

result = [item for item in result if item.strip()]
# print(len(result))
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])
all_words = sorted(set(preprocessed))


# print(all_words)

# print(len(all_words))


vocab = {token:integer for integer, token in enumerate(all_words)}

# print(vocab)

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed_text = re.split(token_pattern, text)

        preprcessed = [
            item.strip() for item in preprocessed_text if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprcessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\']|--)', r'\1', text)
        return text

tokenizer = SimpleTokenizer(vocab)

text = "Poor Jack! It had always been his fate to have women say such things of him: the fact should be set down in extenuation. What struck me now was that, for the first time, he resented the tone. I had seen him, so often, basking under similar tributes--was it the conjugal note that robbed them of their savour? No--for, oddly enough, it became apparent that he was fond of Mrs. Gisburn--fond enough not to see her absurdity. It was his own absurdity he seemed to be wincing under--his own attitude as an object for garlands and incense."


# ids = tokenizer.encode(text)

# print(ids)


# idss = tokenizer.decode(ids)
# print(idss)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab))


# for i, num in enumerate(list(vocab.items())[-5:]):
#     print(i, num)


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

tokenizer_v2 = SimpleTokenizerV2(vocab)

text = "Poor Jack! It had always been his fate to ggs"

ids = tokenizer_v2.decode(tokenizer_v2.encode(text))

# print(ids)

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.decode(tokenizer.encode("Poor Jack! It had always been his fate to ggs")))  

# data sampling


enc_text = tokenizer.encode(raw_text)


enc_sample = enc_text[50:]
print(len(enc_text))
context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(x)
print(y)

import torch

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


input_ids = torch.tensor([[50256,  314,  232,  318]])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)