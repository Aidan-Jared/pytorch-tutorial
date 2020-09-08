import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtextModel import Encoder, Decoder, Attention, Seq2Seq
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=.1)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    SRC = Field(
                tokenize= 'spacy',
                tokenizer_language= 'de',
                init_token= '<sos>',
                eos_token= '<eos>',
                lower= True
    )

    TRG = Field(
                tokenize= 'spacy',
                tokenizer_language= 'en',
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
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())
    print(f'The model has {count_parameters(model):,} trainable parameters')

    PAD_IDX = TRG.vocab.stoi('<pad>')
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)