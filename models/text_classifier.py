from torch import nn
from pytorch_transformers import BertConfig, BertForSequenceClassification

bert_config = BertConfig(
    vocab_size=50000,
    max_position_embeddings=768,
    intermediate_size=2048,
    hidden_size=512,
    num_attention_heads=8,
    num_hidden_layers=6,
    type_vocab_size=5,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_labels=2,
)

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.net = BertForSequenceClassification(bert_config)

    def forward(self, x):
        
        return self.net(x)[0]
