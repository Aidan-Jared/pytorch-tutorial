import torch
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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