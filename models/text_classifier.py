import torch
from torch import nn
import torch.nn.functional as F

class SimpleTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes,
        vocab_size=95811,
        embed_dim=64
    ):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, inputs):
        text = inputs.text
        offsets = inputs.offsets
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class TextCNN(nn.Module):
    def __init__(
        self, 
        num_classes,
        vocab_size=30522, 
        embedding_dim=100, 
        n_filters=100, 
        filter_sizes=[3, 4, 5], 
        dropout=0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1, 
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)