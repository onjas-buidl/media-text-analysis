import os
import codecs
import pandas as pd
import chardet
import pandas as pd
import numpy as np
import jieba as jb
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import gensim
from gensim import corpora
import tqdm

# import pyLDAvis.gensim


# %%
def get_encoding(file):
	with open(file, 'rb') as f:
		data = f.read()
		return chardet.detect(data)['encoding']


def loadDataFromFile(filepath):
	edn = ''
	data = pd.DataFrame()
	i = 0
	print('开始读取文件,请稍等片刻....')
	for fpath, dirs, fs in os.walk(filepath):
		file = fs[0]
		filename = fpath + '/' + file
		encodingname = get_encoding(filename)
		if edn != encodingname:
			edn = encodingname
		print('fin loading coding method')
		for file in fs:
			filename = fpath + '/' + file
			# encodingname = get_encoding(filename)
			# if edn != encodingname:
			# 	edn = encodingname
				# print(encodingname,filename)
			label = [dir for dir in fpath.split('/')][-1]
			# content = codecs.open(filename, "r", "ANSI").read()
			content = codecs.open(filename, "r", edn).read()
			content_table = pd.DataFrame({'text': content, 'label': label, 'file': file}, index=[i])
			data = data.append(content_table)
			i += 1
	print('文件读取完成!')
	return data

#%%
df = loadDataFromFile('dir')

# %%

def remove_punctuation(line):
	# 定义删除除字母,数字，汉字以外的所有符号的函数
	line = str(line)
	if line.strip() == '':
		return ''
	rule = re.compile(u"[^\u4E00-\u9FA5]")
	line = rule.sub('', line)
	return line


# 停用词列表
def stopwordslist(filepath):
	stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
	return stopwords


# 加载停用词
stopwords = stopwordslist("stop_words.txt")
stopwords = stopwords + ['#', '##', '###', '年', '月']
#%% TOKENIZE --very slow, a few minutes
# 删除除字母,数字，汉字以外的所有符号
df['clean_text'] = df['text'].apply(remove_punctuation)

# 分词，并过滤停用词
df['cut_text'] = df['clean_text'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df[['text', 'clean_text', 'cut_text']].head()
print('afin!')

# df[['file', 'cut_text']].to_csv('dir_tokenized.csv')
#%%
no_features = 1000

tfidf = TfidfVectorizer(max_features=no_features)
tfidf_features = tfidf.fit_transform(df.cut_text)
tfidf_feature_names = tfidf.get_feature_names()


cv = CountVectorizer(max_features=no_features)
cv_features = cv.fit_transform(df.cut_text)
cv_feature_names = cv.get_feature_names()

sum([(i in cv_feature_names) for i in tfidf_feature_names])

#%%
no_topics = 9
#NMF
nmf_tfidf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf_features)
#LDA
lda_cv = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(cv_features)
#%%
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("主题 {} : {}".format(topic_idx,"|".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])))

no_top_words = 10
print('---------------NMF-tfidf_features 主题-----------------------------------------')
display_topics(nmf_tfidf, tfidf_feature_names, no_top_words)
print()
print('--------------Lda-CountVectorizer_features 主题--------------------------------')
display_topics(lda_cv, cv_feature_names, no_top_words)

#%%
def predict_topic_by_cv(text):
    txt = remove_punctuation(text)
    txt = [w for w in list(jb.cut(txt)) if w not in stopwords]
    txt = [w for w in txt if len(w)>1]
    txt = [" ".join([w for w in txt])]
    newfeature = cv.transform(txt)
    doc_topic_dist_unnormalized = np.matrix(lda_cv.transform(newfeature))
    doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
    topicIdx = doc_topic_dist.argmax(axis=1)[0,0]
    print('该文档属于:主题 {}'.format(topicIdx))
    print("主题 {} : {}".format(topicIdx,"|".join([cv_feature_names[i] for i in (lda_cv.components_[topicIdx,:]).argsort()[:-no_top_words - 1:-1]])))

#%%
text_data = df.cut_text.apply(lambda x:x.split())
#过滤掉单个汉字的词语
text_data = text_data.apply(lambda x:[w for w in x if len(w)>1] )

dictionary = corpora.Dictionary(text_data)

#过滤掉词频小于5次,和词频大于90%的词语
dictionary.filter_extremes(no_below=5, no_above=0.9)