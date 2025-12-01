import os
import torch


# ****************** begin hyperparameters ******************
train_proportion = 0.9 # proportion of the data that's reserved for training, rest is for validation
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # this many steps of gradient descent
eval_interval = 500 # eval every so often to plot loss as model trains
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # when u do an eval, sample 200 batches so u can report avg loss across those (individual batches can be noisy)
n_embd = 384 # num dims of embedding vectors
n_head = 6
n_layer = 6
dropout = 0.2
# ****************** END hyperparameters ******************





# ****************** BEGIN other global variables ******************
dataset_filename = 'input.txt'
torch.manual_seed(1337) # allows reproducibility
# ****************** END other global variables ******************





# ****************** BEGIN data loading, tokenization, train-val split ****************** 

with open(filepath, 'r', encoding='utf-8') as f:
    return f.read()


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
    
encode = lambda s: stoi[c] for c in s]
decode = lambda l: ''.join([self.itos[i] for i in l])
    
data = torch.tensor(encode(text), dtype=torch.long)


n = int(train_proportion*len(data)) # truncate to a whole number bc index can only be an int
    
    train_data = data[0:n] # first 90% of data will be used to train
    val_data = data[n:] # remaining 10% will be held out for validation
    return train_data, val_data

# ****************** END data loading, tokenization, train-val split ******************





# ****************** BEGIN batching and eval loss helper functions ****************** 

def get_batch(split: str):
    """ generate a batch of data of inputs x and targets y. split param can either be 'train' or 'val'"""
    
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data)-block_size, (batch_size,)) # generate batch_size number of random indices to pull snippets from
    x = torch.stack([data[i:i+block_size] for i in idx]) # torch.stack() takes in a list/tuple of tensors
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # e.g. for x = snippet[0] y = snippet[1], hence +1
    x,y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    """evaluating current model on training and validation sets. spits out avg loss for both in a dictionary w 2 keys"""
    metrics = {}
    model.eval() 
    losses = torch.zeros(eval_iters) # use this instead of list so u can call losses.mean() instead of sum(losses)/len(losses) for a list
    splits = ['train', 'val']
    for split in splits: 
        for i in range(eval_iters):
            x, y = get_batch(key)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        metrics[key] = losses.mean()
        
    model.train() # reset to training mode
    return metrics

# ****************** END batching and eval loss helper functions ****************** 





# ****************** BEGIN define transformer model ****************** 
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # for every token in the vocab (characters in our case) there should be a vector representation
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

# ****************** END define transformer model ****************** 





# ****************** BEGIN initialize model and optimizer ****************** 

model = Transformer()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ****************** END initialize model and optimizer ****************** 





# ****************** BEGIN training loop w periodic evals to gauge progress ****************** 

for step in range(max_iters):
    if step % eval_interval == 0 or iter == max_iters - 1: # check if its time for the periodic eval
        losses = estimate_loss()
         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    
    optimizer.zero_grad(set_to_none=True) # clear gradients from previous training step
    
    x, y = get_batch('train') # sample new batch
    
    logits, loss = model(x, y) # forward pass
    loss.backward() # calculate gradients
    optimizer.step() # step in the direction of steepest descent

# ****************** END training loop w periodic evals to gauge progress ****************** 


    


# ****************** BEGIN generate from final model ****************** 

context = torch.zeros((1, 1), dtype=torch.long, device=device) #First dimension (1): Batch size - generating 1 sequence at a time. Second dimension (1): Sequence length - starting with just 1 token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# ****************** END generate from final model ****************** 

