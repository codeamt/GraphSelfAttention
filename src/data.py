import numpy as np
import re 
from typing import List, Dict
from collections import Counter

def synth_paragraph(corpus, length=5):
    '''
    Generates a synthetic paragraphg using Markov chain.
    Args:
    - corpus (List[str]): list of seed sentences for generating new paragraphs. 
    - length (int): number of sentences in new paragraph.
    Returns:
    - paragraph (str): new augmented example 
    '''
    words = [word for sentence in corpus for word in sentence.split()]
    pairs = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    def get_transition_probs(pairs):
        probs = {}
        for w1, w2 in pairs:
            if w1 in probs:
                probs[w1].append(w2)
            else:
                probs[w1] = [w2]
        return probs
    transition_probs = get_transition_probs(pairs)
    def gen_paragraph(seed, probs, words):
        output = []
        seed = np.random.choice(words)
        for _ in range(length):
            output.append(seed)
            if seed in probs:
                next_word = np.random.choice(probs[seed])
                seed = next_word
            else:
                seed = np.random.choice(words)
        return ' '.join(output)
    
    paragraph = gen_paragraph(transition_probs, words)
    return paragraph
    

def prepreocess(paragraph):
    '''
    Prepares paragraph for NLP tasks.
    Args: 
    - paragraph (str): Input paragraph of any length. 
    Returns:
    - cleaned (List[str]): list of preprocessed sentences 
    '''
    sentences = re.split(r'<regex>', paragraph)
    cleaned = []
    for sent in sentences:
        clean_sent = re.sub(r'<regex>', '', sent).lower()
        cleaned.append(clean_sent)
    return cleaned 

def get_word_index(sentences):
    '''
    Maps words to indexes for given list of preprocessed sentences
    Args: 
    - sentences (List[str]): cleaned sentences.
    Returns:
    - words_to_index (Dict[str:int]): word-index map. 
    '''
    words_to_index = { word: idx for idx, word in enumerate(set(' '.join(sentences).split())) }
    return words_to_index


def sentence_to_tensor(sentences, word_to_index, max_length=10):
    '''
    Converts sentences to tensor inputr representations.
    Args:
    - sentences (List[str]): preprocessed sentence
    - word_to_index (Dict[str:int]): word mapping
    - max_length (int): max length of each sentence. 
    Returns: 
    - tensor_output (numpy array): tensor representation of sentences
    '''
    tensor_output = np.zeros((len(sentences), max_length), dtype=np.int32)
    for i, sent in enumerate(sentences):
        words = sent.split()[:max_length]
        for j, word in enumerate(words):
            tensor_output[i, j] = word_to_index[word]
    return tensor_output


def get_adj_matrix(tensor_input):
    '''
    Generates adjacency graph for given tensor input.

    Args:
    - tensor_input (numpy array): tensor representation of sentences
    Returns:
    - adj_matrix (numpy array): relationships between sentences. 
    '''
    adj_matrix = np.zeros((len(tensor_input), len(tensor_input)), dtype=np.int32)
    for i in range(len(tensor_input)):
        for j in range(len(tensor_input)):
            if i != j:
                if np.sum(tensor_input[i] == tensor_input[j]) > 0:
                    adj_matrix[i, j] = 1
    return adj_matrix