#!usr/bin/env python
# -*- coding: utf-8 -*-

# set default coding euc-kr 2 utf-8
import sys
reload(sys)

sys.setdefaultencoding('utf-8')

print ("load")
#load from kobill 
#from konlpy.corpus import kobill
#docs_ko =kobill.open('kobill/news.txt').read()

print ("tokenize")
#tokenize
from konlpy.tag import Twitter; t = Twitter()
print ("tokenize1")
pos = lambda d:['/'.join(p) for p in t.pos(d,stem=True,norm=True)]
print 'tokenize2'
texts_ko = pos('나쁜 고양이가 사료를 먹었습니다.\n그래서 내가 뺏어먹었습니다.')
print texts_ko
print ("train")
import time
now_time = time.time()
#train
from gensim.models import word2vec
#wv_model_ko = word2vec.Word2Vec(texts_ko,workers=4)
wv_model_ko = word2vec.Word2Vec(texts_ko,workers=16,negative=10,window=7,size=300)
wv_model_ko.init_sims(replace=True)
print 'train end'

wv_model_ko.save('ko_word2vec_e.model')
print wv_model_ko.vocab
print ("training time "+str(time.time()-now_time)+"sec")
print wv_model_ko.most_similar(u'o')

