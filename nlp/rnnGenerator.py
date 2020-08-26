from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

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
    print(unicodeToAscii("O'Néàl"))