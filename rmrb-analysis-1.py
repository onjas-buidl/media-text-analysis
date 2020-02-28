#%%
import jieba
import jieba.posseg as pseg
import collections
import nltk
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
#%% defining gini functions
# def gini_1(x):
#     # (Warning: This is a concise implementation, but it is O(n**2)
#     # in time and memory, where n = len(x).  *Don't* pass in huge
#     # samples!)
#     # Mean absolute difference
#     mad = np.abs(np.subtract.outer(x, x)).mean()
#     # Relative mean absolute difference
#     rmad = mad/np.mean(x)
#     # Gini coefficient
#     g = 0.5 * rmad
#     return g

def gini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


#%% Timing
"""
collection.Counter is slower than pd.Series.value_counts(), consistently
"""

X = np.random.rand(10000)

X = list(X)
t = time.time()
# filter(lambda x: x > .5, X)
collections.Counter(words)
print(time.time() - t)

t = time.time()
# filter(lambda x: x > .5, X)
pd.Series(words).value_counts()
print(time.time() - t)






#%% get the file path
import os
path = 'working_data/'
month_folders = [i for i in os.walk(path)][0][1]
len(month_folders)
#%% stop words
with open('stop_words.txt', 'r') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')
stop_words = stop_words + ['#', '##', '###']
stop_words = np.array(stop_words)

#%%

text = ''
word_count_collection = {}
gini_collection = {}
num_word_collection = {}
# for each month, compute its statistics
for m in tqdm.tqdm(month_folders):
    articles = [i for i in os.walk(path + m + r'/')][0][2]
    for md_f in articles:
        with open(path + m + r'/' + md_f, 'r') as file:
            text += file.read().replace('\n', '').replace(' ', '')
    seg_list = jieba.cut(text, cut_all=False)
    words = np.array([i for i in seg_list])
    # words = filter(lambda x: x not in stop_words, words)
    words = words[np.isin(words, stop_words, invert=True)]
    hist_table = pd.Series(words).value_counts()
    g = gini(hist_table)
    print(m, str(len(words)), 'unique words')
    print(m, 'Gini coeff:', str(g))
    num_word_collection[m] = len(words)
    word_count_collection[m] = hist_table
    gini_collection[m] = g

#%% data visulizations
for key in sorted(gini_collection.keys()):
    print("%s: %s" % (key, gini_collection[key]))


L = sorted(gini_collection.items()) # sorted by key, return a list of tuples

x, y = zip(*L) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()


#%%

articles = [i for i in os.walk(path + month_folder + '/')][0][2]
month_text = ''
for a in articles:
    with open(month_folder + r'/' + a) as file:
        month_text += file.read().replace('\n', '').replace(' ', '')
jieba_obj = jieba.cut(month_text, cut_all=False)
words = np.Series([i for i in seg_list])
print(month_folder, 'word num: ', str(len(words)))
count = words.value_counts()

#%%

c = collections.Counter(res)

#%%

labels, values = zip(*c.items())

#%%

len(values)

#%%

sorted_val = list(values)
sorted_val.sort(reverse=True)

#%%

plt.scatter([i for i in range(len(sorted_val))], sorted_val)

#%%



#%%

val_array = np.array(values)

#%%

gini(val_array)

#%%


