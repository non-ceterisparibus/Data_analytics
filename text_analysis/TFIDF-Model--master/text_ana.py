import nltk
import re
import heapq # to find n most frequent words in dictionary
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from collections import Counter
from string import punctuation
import wordcloud
import matplotlib.pyplot as plt

# Tokenize Vietnamese
def VietnameseTokenizer(comments):
    list_comments = []
    for comment in comments:
        cmt_str = comment.to_string()
        text_lower = cmt_str.lower()
        text_token = ViTokenizer.tokenize(text_lower)
        list_comments.append(text_token)
    return list_comments

def remove_stop_words(comments, stop_word):
    sentences = []
    for comment in comments:
        for word in comment.split(" "):
            if (word not in stop_word):
                if ("_" in word) or (word.isalpha() == True):
                    sentences.append(word)
    return sentences

def wordfreq(sentence):
    word_freq = {}
    for word in sentence:
        # print(word)
        if word in word_freq:
            word_freq[word] += 1 # count
        else:
            word_freq[word] = 1
    return word_freq

def word_mostcount(word_freq):
    # to find the most frequent words in dictionary
    most_freq_words = []
    max_freq = max(word_freq.values())
    for word, freq in word_freq.items():
        if freq == max_freq:
            most_freq_words.append(word)

    # freq_word = heapq.nlargest(100, word2count, key = word2count.get)
    return most_freq_words

def n_mostcount(word_freq,n):
    # Sort the words by frequency in descending order
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Get the 100 most frequent words
    most_freq_words = sorted_words[:n]

    return most_freq_words



def wordidfs(freq_word, sentences):
    # TF Matrix
    word_idfs = {}
    for word in freq_word:
        doc_count = 0
        for word in sentences:
            if word in ViTokenizer.tokenize(word):
                doc_count += 1
        word_idfs[word] = np.log((len(sentences)/doc_count)+1)
    return word_idfs

def tfmatrix(freq_word, sentences):
    tf_matrix = {}
    for word in freq_word:
        doc_tf = []
        for word in sentences:
            frequency = 0
            for w in ViTokenizer.tokenize(word):
                if word == w:
                    frequency += 1
            tf_word = frequency/len(nltk.word_tokenize(word))
            doc_tf.append(tf_word)
        tf_matrix[word] = doc_tf

    return tf_matrix

def tf_idfmatrix(tf_matrix,word_idfs):
    tfidf_matrix = []
    for word in tf_matrix.keys():
        tfidf = []
        for value in tf_matrix[word]:
            score = value * word_idfs[word]
            tfidf.append(score)
        tfidf_matrix.append(tfidf)       
                
    x = np.asarray(tfidf_matrix)  
    x = np.transpose(x)

    return x

def cloudmask(word_freq,):
    # read the mask image
    cloud_mask = np.array(Image.open("round.jpg"))

    # create wordcloud
    wc = wordcloud.WordCloud(max_words=NUM_WORDCLOUD,  
                             background_color="white", colormap="viridis",
                             font_path=f"{path}TakaoMincho.ttf",
                             mask=cloud_mask, 
                             prefer_horizontal=1,
                             mode="RGB").generate_from_frequencies(word_freq)

    plt.figure(figsize=(40, 20))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('wordcloud.png')

def wordcloud(sentences):
    cloud = np.array(sentences).flatten()
    plt.figure(figsize=(20,10))
    word_cloud = wordcloud.WordCloud(max_words=100,background_color ="black",
                                   width=2000,height=1000,mode="RGB").generate(str(cloud))
    plt.axis("off")
    plt.imshow(word_cloud)