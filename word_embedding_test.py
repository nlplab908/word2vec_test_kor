#!usr/bin/env python
# -*- coding: utf-8 -*-

# set default coding euc-kr 2 utf-8
import sys
reload(sys)

sys.setdefaultencoding('utf-8')

print ("load")
#load from kobill 
from konlpy.corpus import kobill
#docs_ko =kobill.open('kobill/news.txt').read()
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]
print ("tokenize")

#tokenize
from konlpy.tag import Twitter; t = Twitter()
print ("tokenize1")
pos = lambda d:['/'.join(p) for p in t.pos(d,stem=True,norm=True)]
print ('tokenize2')
texts_ko = [pos(doc) for doc in docs_ko]
#texts_ko = pos(docs_ko)
print ("train")
import time
now_time = time.time()
#train
from gensim.models import word2vec
wv_model_ko = word2vec.Word2Vec(texts_ko,workers=16,negative=10,window=7,size=300)
wv_model_ko.init_sims(replace=True)


wv_model_ko.save('ko_word2vec_e.model')
print ("training time "+str(time.time()-now_time)+"sec")
print (wv_model_ko.most_similar((pos('서울대학교'))))
