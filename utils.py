from Seq2Seq import Seq2Seq
import torch.nn as nn
import torch
import numpy as np
import re
from collections import Counter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Train the model for 1 epoch
def epoch(train_loader, val_loader, model, criterion, optimizer):

    train_loss = 0
    model.train()
    for batch_num, (input, target) in enumerate(train_loader, 1):

        output = model(input, target[:,:-1])                    
        loss = criterion(output.flatten(0, 1), target[:, 1:].flatten())      
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_num % 10000 == 0):
            print(f"Batch : {batch_num} Loss : {loss.item() / input.size(0)}")

    print(f"Training Loss : {train_loss/len(train_loader.dataset)}")

    val_loss = 0
    model.eval()
    for input, target in val_loader:

        output = model(input, target[:,:-1])                    
        loss = criterion(output.flatten(0, 1), target[:, 1:].flatten())      
        val_loss += loss.item()
    
    print(f"Validation Loss : {val_loss/len(val_loader.dataset)}")


# Choose the one with that maximum probability
def greedy_decode(model, input, vocab, tgt_len):

    #Reduce memory usage and speed up computations
    with torch.no_grad():     
        
        # encoder_output: [ batch_size, max_len, 2 * hidden_size ]
        # encoder_final:  [ 1, batch size, 2 * hidden_size ]
        encoder_output, encoder_final = model.encode(input = input)

    # Store generated output
    generated = []

    # Initialize with sos
    prev_output = torch.ones(1, 1).fill_(vocab['<bos>']).type_as(input)

    decoder_hidden = None
        
    for t in range(tgt_len):
        with torch.no_grad():
            
            output, decoder_hidden = model.decode(target = prev_output, 
            encoder_output = encoder_output, encoder_final = encoder_final, 
            decoder_hidden = decoder_hidden)

            # probs: [ batch_size, vocab_size ]
            probs = model.generator(output[:, -1])

        # Greedily pick next word
        _, next_word = torch.max(probs, dim=1)
        next_word = next_word.data.item()

        # Sample next word
        # p = probs.squeeze(0).detach().cpu().numpy().astype('float64')
        # next_word = np.random.multinomial(1, np.exp(p) / sum(np.exp(p)), 1).argmax()
        
        generated.append(next_word)
        prev_output = torch.ones(1, 1).type_as(input).fill_(next_word) 

    generated = np.array(generated)

    # Cut off everything starting from first_eos
    first_eos = np.where(generated == vocab['<eos>'])[0]
    if( len(first_eos) > 0 ):
        generated = generated[:first_eos[0]]

    return generated
    

def validation_examples(model, test_loader, vocab, noOfExamples=3, tgt_len=15):
    
    model.eval()
    samples = []
        
    for batch_num, (input, target) in enumerate(test_loader, 1):
      
        # Assume batch_size = 1, take the first example of each batch
        result = greedy_decode(model, input.to(device), vocab, tgt_len)

        input = input.cpu().numpy()[0, :]
        target = target[:,1:].cpu().numpy()[0, :]  
        samples.append((input, target, result))

        if batch_num == noOfExamples:
            break

    return samples

def reconstruction(tokens, vocab, return_tokenized=False):
    tokens = tokens.tolist()
    try:
        first_eos = tokens.index(vocab.stoi['<eos>'])
    except ValueError:
        first_eos = len(tokens)
    tokens = tokens[:first_eos]
    specials=['<unk>', '<pad>', '<bos>', '<eos>']

    if return_tokenized:
        tokens = [token for token in tokens if vocab.itos[token] not in specials]
    else:
        tokens = " ".join([vocab.itos[token] for token in tokens 
                if vocab.itos[token] not in specials])
    
    return tokens

def get_reconstructed(samples, vocab, return_tokenized=False):
    targets, results = [], []
    for num, sample in enumerate(samples):
        _, target, result = sample
        target_rec = reconstruction(target, vocab, return_tokenized)
        result_rec = reconstruction(result, vocab, return_tokenized)
        if(len(result_rec) > 0 and len(target_rec) > 0):
            targets.append(target_rec)
            results.append(result_rec)
    return targets, results

def train(vocab, embedding, train_loader, val_loader, epochs = 100, lr = 0.0003):
    
    model = Seq2Seq(embedding = embedding, hidden_size = 256, 
    vocab_size = len(vocab)).to(device)
    
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=vocab['<pad_idx>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch_num in range(1, epochs+1):
        try:
            print("\nEpoch %d" % epoch_num)
            epoch(train_loader, val_loader, model, criterion, optimizer)
            torch.save(model.state_dict(), f'model{epoch_num}')
        except KeyboardInterrupt:
            return model

def tokenize(data, vocab):
    tokenized  = []
    for s, t in data:
        s_tok = [vocab[token] for token in s.split() if vocab[token] != vocab['<unk>']]
        t_tok = [vocab[token] for token in t.split() if vocab[token] != vocab['<unk>']]
        tokenized.append((s_tok, t_tok))
    return tokenized

def buildCounter(data):
    counter = Counter()
    for s, t in data:
        if (len(t.split()) > 0):
                counter.update(s.split())
                counter.update(t.split())
    return counter

# Lowercase, trim, and remove non-letter characters
def normalize(line):
        return re.sub(r"[^a-zA-Z]+", r" ", line.lower())

def extractData(reviews, summaries):
    data = []
    for s, t in zip(reviews, summaries):
        if t is np.nan:                            #Some summaries are nan
            continue
        data.append((normalize(s), normalize(t)))
    return data


def fetch_grams(tokens, n):
    return [tokens[i-n+1 : i+1] for i in range(n-1, len(tokens))]

def ngram_similarity(n_r, n_t, similarity):
    return sum([similarity[tokens] for tokens in zip(n_r, n_t)]) / len(n_r)

def bleu_similarity(result, target, similarity):
    
    if(not len(result)):
        return (not len(target)) * 1.0
    
    precisions = []                       #Precisions for all grams
    
    #Handle Unigrams
    precision = 0                         #Overall precision for unigrams
    for uni_r in result:
        
        p_single = [0]                    #Precision for single unigram
        for uni_t in target:
            p_single.append(similarity[uni_r, uni_t])
        precision += max(p_single)
    precision /= len(result)
        
    if precision > 0:
        precisions.append(precision)
        
        
    #Handle n-grams
    for n in range(2, 5):
        
        if(len(result) < n):
            break
        
        n_target = fetch_grams(target, n)
        n_result = fetch_grams(result, n)
        precision = 0                         #Overall precision for n-grams
        
        for n_r in n_result:
            p_single = [0]                    #Precision for single n=gram

            for n_t in n_target:
                p_single.append( ngram_similarity(n_r, n_t, similarity) ) 
            precision += max(p_single)
            
        precision /= len(n_result)
        if precision > 0:
            precisions.append(precision)
           
    
    if (len(result) > len(target)):
        BP = 1
    else:
        BP = np.exp(1 - (len(target)/len(result)))
    
    return np.exp(np.log(precisions).mean()) * BP