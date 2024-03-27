import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split   
from torchtext.data.functional import to_map_style_dataset

from .constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a subclass of tuple on which calling .cuda() will call .cuda() on each element
class CudaTuple(tuple):

    def cuda(self):
        return tuple(map(lambda x: x.cuda(), self))

def load_ag_news(batch_size=64):

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return CudaTuple(text_list.to(device), offsets.to(device)), label_list.to(device)

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    train_iter = AG_NEWS(split="train", root=AG_NEWS_PATH)
    test_iter = AG_NEWS(split="test", root=AG_NEWS_PATH)

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    train_ds = to_map_style_dataset(train_iter)
    num_train = int(len(train_ds) * 0.95)
    split_train_, split_valid_ = random_split(
        train_ds, 
        [num_train, len(train_ds) - num_train],
        generator=torch.Generator().manual_seed(DATASET_SPLIT_SEED)
    )
    train_dl = DataLoader(
        split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_dl = DataLoader(
        split_valid_, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_ds = to_map_style_dataset(test_iter)
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    return train_dl, val_dl, test_dl

if __name__ == "__main__":

    test = CudaTuple((torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])))
    test = test.cuda()
    print(test)
    breakpoint()
    