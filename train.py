import datetime
import time

from seqeval.metrics import f1_score, classification_report
from torch import optim
from tqdm import tqdm

from data_loader import ConllDataset, get_data_loader
from model import CRF, NER
from parameters import params
from utils import *

assert torch.cuda.is_available()

# hyper parameters
batch_size = params['batch_size']
max_epoch = params['max_epoch']
feature_size = params['feature_size']
feature_dim = params['feature_dim']
learning_rate = params['learning_rate']
word_embed_dim = params['word_embed_dim']
char_embed_dim = params['char_embed_dim']
hidden_dim = params['hidden_dim']
dropout_p = params['dropout_p']
grad_clip = params['grad_clip']
seed = params['seed']
device_id = params['device_id']
resume = params['resume']

fix_seed(seed)
torch.cuda.set_device(device_id)

cur_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
logger = get_logger('./results/logs/' + cur_time + '.log')

for item in params:
    logger.info(f'{item} = {params[item]}')

start_time = time.time()

vocab = load_file('./data/vocab.json')
train_data_loader = get_data_loader(dataset=ConllDataset('./data/dataset/train.json', vocab),
                                    batch_size=batch_size,
                                    shuffle=True)
val_data_loader = get_data_loader(dataset=ConllDataset('./data/dataset/valid.json', vocab),
                                  batch_size=batch_size,
                                  shuffle=False)
test_data_loader = get_data_loader(dataset=ConllDataset('./data/dataset/test.json', vocab),
                                   batch_size=1,
                                   shuffle=False)

word_size = len(vocab['word2id'])
char_size = len(vocab['char2id'])
tag_size = len(vocab['tag2id'])

id2tag = {key: value for value, key in vocab['tag2id'].items()}

logger.info(f'data processing consumes: {(time.time() - start_time):.2f}s')

model = NER(CRF(tag_size), word_size, char_size, feature_size, feature_dim,
            word_embed_dim, char_embed_dim, hidden_dim, dropout_p).cuda()
model.apply(weights_init)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stop = EarlyStopping(mode='max', min_delta=0.01, patience=3)

if resume:
    checkpoint, cp_name = load_checkpoint(latest=True)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info(f'load checkpoint: [{cp_name}]')


def train():
    model.train()
    epoch_loss = 0
    for s, t, c, l in tqdm(train_data_loader):
        # s, t, c, l: means sentences, tags, chars, batch_len, respectively
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(s, t, c, l)
        loss.backward()
        epoch_loss += loss.item()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return epoch_loss / len(train_data_loader)


def validate():
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for s, t, c, l in val_data_loader:
            loss = model.neg_log_likelihood(s, t, c, l)
            epoch_loss += loss.item()
    return epoch_loss / len(val_data_loader)


def evaluate():
    model.eval()
    score_eval = 0.0
    y_true_eval, y_pred_eval = [], []
    with torch.no_grad():
        for s, t, c, l in test_data_loader:
            pred_score, pred_tag = model(s, c, l)
            score_eval += pred_score.item()
            y_true_eval.append(translate(id2tag, t.cpu()))
            y_pred_eval.append(translate(id2tag, pred_tag.cpu()))
        score_eval /= len(test_data_loader)
    return score_eval, y_true_eval, y_pred_eval


for epoch in range(1, int(max_epoch + 1)):
    start_time = time.time()

    train_loss = train()
    val_loss = validate()
    score, y_true, y_pred = evaluate()

    f1 = f1_score(y_true=y_true, y_pred=y_pred) * 100

    epoch_min, epoch_sec = record_time(start_time, time.time())
    logger.info(
        f'epoch: [{epoch:02}/{max_epoch}]  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  '
        f'score={score:.2f}  f1={f1:.2f}  duration: {epoch_min}m {epoch_sec}s')

    if early_stop.step(f1):
        logger.info(f'early stop at [{epoch:02}/{max_epoch}]')
        logger.info(classification_report(y_true=y_true, y_pred=y_pred))
        break

save_checkpoint(experiment_time=cur_time, model=model, optimizer=optimizer)

logger.info('training finished.')
