from torch import nn

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
        text, offsets = inputs
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)