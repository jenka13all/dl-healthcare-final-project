import torch
import model.net as net
import csv

checkpoint_file = 'checkpoints/best_model.pt'
vocab_file = 'data/cohort-vocab.csv'

# model parameters
model_param = {'num_epochs': 5,
               'batch_size': 128,
               'embedding_size': 100,
               'kernel_size': 5,
               'learning_rate': 0.0001,
               'weight_decay': 1e-5
               }

# length of padded sub-sequences
len_padded = 32


def get_vocab(vocab_file):
    with open(vocab_file) as f:
        rd = csv.reader(f)
        next(rd)
        vocab = {}
        for r in rd:
            vocab[int(r[0])] = r[3]
        vocab_size = len(vocab) + 1

    return vocab, vocab_size


vocab, vocab_size = get_vocab(vocab_file)

model = net.ehrEncoding(vocab_size=vocab_size,
                        max_seq_len=len_padded,
                        emb_size=model_param['embedding_size'],
                        kernel_size=model_param['kernel_size'],
                        pre_embs=None,
                        vocab=vocab)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=model_param['learning_rate'],
                             weight_decay=model_param['weight_decay'])

# load saved model
checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
