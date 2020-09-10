import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtextModel import Encoder, Decoder, Attention, Seq2Seq
from Transformer import Transformer

import math
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=.1)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _,batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _,batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    SRC = Field(
                tokenize= 'spacy',
                tokenizer_language= 'de_core_news_sm',
                init_token= '<sos>',
                eos_token= '<eos>',
                lower= True
    )

    TRG = Field(
                tokenize= 'spacy',
                tokenizer_language= 'en_core_web_sm',
                init_token= '<sos>',
                eos_token= '<eos>',
                lower= True
    )

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
    
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
                                                                        (train_data, valid_data, test_data),
                                                                        batch_size= BATCH_SIZE,
                                                                        device=device
    )

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    # ENC_EMB_DIM = 32
    # DEC_EMB_DIM = 32
    # ENC_HID_DIM = 64
    # DEC_HID_DIM = 64
    # ATTN_DIM = 8
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5

    # enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    # attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    # dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    # model = Seq2Seq(enc, dec, device).to(device)
    # model.apply(init_weights)
    d_model = 512
    heads = 8
    N = 1
    SRC_PAD_IDX = SRC.vocab.stoi['<pad>']
    TRG_PAD_IDX = TRG.vocab.stoi['<pad>']

    model = Transformer(INPUT_DIM, OUTPUT_DIM, d_model, N, heads, SRC_PAD_IDX, TRG_PAD_IDX)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = optim.Adam(model.parameters(), lr=.0001, betas=(.9,.98), eps=1e-9)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')