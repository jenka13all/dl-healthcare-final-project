from model.data_loader import EHRdata, ehr_collate
import evaluate_for_predictions as evaluate
from torch.utils.data import DataLoader
import model.net as net
import torch
import utils as ut
import vocabulary
import os

sampling = None
emb_filename = None
embs = None

indir = './data'

exp_dir = os.path.join(indir, 'encodings', 'test')
os.makedirs(exp_dir, exist_ok=True)

# get the vocabulary size
vocab_size, vocab = vocabulary.get_vocab(indir)

# set random seed for experiment reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# load data
data_ts = EHRdata(os.path.join(indir), ut.dt_files['ehr-file-test'], sampling)
data_generator_ts = DataLoader(
    data_ts,
    ut.model_param['batch_size'],  # may need to reduce this
    shuffle=False,
    collate_fn=ehr_collate
)
print("Test cohort size: {0}".format(len(data_ts)))

# define model and optimizer
model = net.ehrEncoding(
    vocab_size=vocab_size,
    max_seq_len=ut.len_padded,  # 32
    emb_size=ut.model_param['embedding_size'],  # 100
    kernel_size=ut.model_param['kernel_size'],  # 5
    pre_embs=embs,
    vocab=vocab
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=ut.model_param['learning_rate'],
    weight_decay=ut.model_param['weight_decay']
)

model.cuda()

loss_fn = net.criterion

# use model from checkpoint
checkpoint = torch.load(os.path.join(indir, 'encodings', 'best_model.pt'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print('\nEvaluating the model from checkpoint')
metrics_avg = evaluate.evaluate_for_predictions(
    model,
    loss_fn,
    data_generator_ts,
    net.metrics
)

# metrics (loss and accuracy)
outfile = os.path.join(exp_dir, 'test_metrics.txt')
with open(outfile, 'w') as f:
    f.write('Mean loss: %.3f\n' % metrics_avg['loss'])
    f.write('Accuracy: %.3f\n' % metrics_avg['accuracy'])
