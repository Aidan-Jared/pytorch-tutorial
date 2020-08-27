from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import time
import math
import unicodedata
import string
import random
import torch
import torch.nn as nn
from rnnGenModelGRU import RNN
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1,n_categories)
    tensor[0][li] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # eos
    return torch.LongTensor(letter_indexes)

def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def train(category_tensor, input_line_tensor, target_line_tensor, model):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()

    model.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    
    loss.backward()
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha = -learning_rate)
    
    return output, loss.item() / input_line_tensor.size(0)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sample(category, model, start_letter="A"):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_lenght):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break # eos
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
    return output_name

def samples(category, model, start_letters = 'ABC'):
    for start_letter in start_letters:
        print(sample(category, model, start_letter))

if __name__ == "__main__":
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1 # for eos

    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    
    n_categories = len(all_categories)
    if n_categories == 0:
        raise RuntimeError('Data not found.')
    print('# categories:', n_categories, all_categories)
    # print(unicodeToAscii("O'Néàl"))
    
    # inputs = category (hot encoded), current letter(hot encoded), hidden state
    # output = next letter, next hidden state

    criterion = nn.NLLLoss()
    learning_rate = .0005

    n_hidden = 128
    model = RNN(n_categories, n_letters, n_hidden, n_letters)
    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample(), model)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.plot(all_losses)

    max_lenght = 20
    
    samples('Russian', model, 'RUS')

    samples('German', model, 'GER')

    samples('Spanish', model, 'SPA')

    samples('Chinese', model, 'CHI')