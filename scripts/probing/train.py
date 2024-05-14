import random, time, os, pytz, argparse, yaml, sys
sys.path.append("../")
import numpy as np
from probing.probe import *
from tqdm import tqdm, trange
from probing_utils import *
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from datetime import datetime
timezone = pytz.timezone('America/New_York') 
date = datetime.now(timezone).strftime("%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default="configs.yaml")
args = parser.parse_args()

# To ensure we get reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def train(model, dataloader, optim, criterion, device):
    
    model.train()
    epoch_loss, epoch_acc = 0, 0
    #print("model dtype = ", next(model.parameters()).dtype)
    for (texts, labels) in tqdm(dataloader, desc="Training", mininterval=10):
        
        optim.zero_grad()
        
        output = model(list(texts)) # batch_size*2, num_classes

        gth = torch.cat([labels, 1 - (labels % 2) + 2*(labels // 2)], dim=1).long().view((-1,)).to(device)
        #print(texts)
        #print(gth)
        #print(f"gth.size() = {gth.size()}")
        #exit()
        #print(f"output.size() = {output.size()}")

        loss = criterion(output, gth)
        
        acc = get_acc(output, gth)

        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

def val(model, dataloader, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0

    with torch.no_grad():
        for (texts, labels) in tqdm(dataloader, desc="Validating", mininterval=10):
            
            output = model(list(texts)) # batch_size*2, num_classes

            gth = torch.cat([labels, 1 - (labels % 2) + 2*(labels // 2)], dim=1).long().view((-1,)).to(device)
        
            loss = criterion(output, gth)
            acc = get_acc(output, gth)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

""" Load Configs """
with open(args.config_file, 'rb') as f: config = yaml.full_load(f.read()) # dict
#os.mkdir(os.path.join(config["save_dir"], date))
#yaml.dump(config, open(os.path.join(config["save_dir"], date, "configs.yaml"), "w"))

""" Prepare Data """
Data = Texts(
    relations=config['relations'].split(),
    nouns_file=config['nouns_file'],
    whoseroles=config['whoseroles'],
)
ood_test_data = Texts(
    relations=config['relations'].split(),
    nouns_file=config['ood_nouns_file'],
    whoseroles=config['whoseroles'],
)

assert config['heldout_ratio'] < 1

num = len(Data)
train_num, heldout_num = num-int(num*config['heldout_ratio']), int(num*config['heldout_ratio'])
train_data, heldout_data = random_split(
    Data,
    [train_num, heldout_num]
)
val_num, test_num = heldout_num-int(heldout_num*0.5), int(heldout_num*0.5)
val_data, test_data = random_split(
    heldout_data,
    [val_num, test_num]
)


print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(val_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(f'Number of ood testing examples: {len(ood_test_data)}')


train_dl = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])
val_dl = DataLoader(val_data, shuffle=True, batch_size=config['batch_size'])
test_dl = DataLoader(test_data, shuffle=True, batch_size=config['batch_size'])
ood_test_dl = DataLoader(ood_test_data, shuffle=True, batch_size=config['batch_size'])

print("Prepare Data: finish\n")


""" Prepare Model """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PROBE(
    hidden_dim_multipliers = config['hidden_dim_multipliers'], 
    num_classes = len(config['relations'].split()), 
    lm = config['lm'], 
    lm_kwargs = config['lm_kwargs'],
    lm_howto_select_encoding_positions = config['lm_howto_select_encoding_positions'], 
).to(device)

optim = Adam(model.parameters(), lr = float(config['lr']))
criterion = nn.CrossEntropyLoss().to(device)

print("Trainable Params: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))
print("Prepare Model: finish\n")


""" Training """
best_val_loss = float('inf')
best_val_acc = 0.0

for epc in trange(config['epochs']):
    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_dl, optim, criterion, device)
    val_loss, val_acc = val(model, val_dl, criterion, device)
    #test_loss, test_acc = val(model, ood_test_dl, criterion, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(config["save_dir"], config["date"], "model.pt"))
    
    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epc+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
    
    #print(f'\tOOD Test Loss: {test_loss:.3f} | OOD Test Acc: {test_acc*100:.2f}%')

print("Training: finish\n")


""" Testing """

model.load_state_dict(torch.load(os.path.join(config['save_dir'], config["date"], "model.pt")))
print("Load ckpt with the best val_loss")

test_loss, test_acc = val(model, test_dl, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

test_loss, test_acc = val(model, ood_test_dl, criterion, device)
print(f'OOD Test Loss: {test_loss:.3f} | OOD Test Acc: {test_acc*100:.2f}%')

print(f"{config['date']}: finish")
