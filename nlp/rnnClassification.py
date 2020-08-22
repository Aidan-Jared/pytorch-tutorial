from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random

import torch
import torch.nn as nn
from rnnClassModel import RNN

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor, model):
    hidden = model.initHidden()
    model.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

if __name__ == "__main__":
    # reading in the data

    print(findFiles('data/names/*.txt'))
    all_letters = string.ascii_letters + '.,;'
    n_letters = len(all_letters)

    category_lines = {}
    all_categories = []

    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    print(category_lines['Italian'][:5])

    # print(letterToTensor('J'))
    # test = lineToTensor("Jones")
    # print(test, test.size())

    n_hidden = 128
    model = RNN(n_letters, n_hidden, n_categories)

    # input = letterToTensor('Albert')
    # hidden = model.initHidden()

    # output, next_hidden = model(input, hidden)
    # print(output)
    # print(categoryFromOutput(output))

    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()
    learning_rate = .005