import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and len(candidate.split()) == 1:
                synonyms.add(candidate)
    return list(synonyms)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    tokens = word_tokenize(example["text"])
    transformed_tokens = []
    replace_prob = 0.30  
    # 30% chance to replace word with synonym

    for token in tokens:
        if token.isalpha() and random.random() < replace_prob:
            syns = get_synonyms(token.lower())
            if syns:
                chosen = random.choice(syns)
                if token.isupper():
                    chosen = chosen.upper()
                elif token.istitle():
                    chosen = chosen.capitalize()
                transformed_tokens.append(chosen)
            else:
                transformed_tokens.append(token)
        else:
            transformed_tokens.append(token)

    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
