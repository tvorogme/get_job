
# coding: utf-8

# In[130]:

import numpy as np
import pickle
import sklearn.feature_extraction.text as text_a
import re
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm


# In[131]:

lemmatizer = WordNetLemmatizer()


# In[132]:

texts = pickle.load(open("answer.pck", "rb"))


# In[133]:

coursera = pickle.load(open("coursera.p", "rb"))


# In[134]:

coursera_texts = [coursera[i]["description"] for i in coursera]


# In[135]:

def clear_text(text):
    global lemmatizer
    
    tmp_text = ""
    
    text = text.lower().replace("\n", " ")
    
    for char in "!@#$%^&*()[]{};:,./<>?|`~-=_+●\\":
        text = text.replace(char, " ")
        
    text = re.sub('[^A-Za-z]+ ', '', text).split(" ")
    
    for word in text:
        tmp_text += lemmatizer.lemmatize(word, 'v') + " "
    
    return tmp_text


# In[136]:

texts = list(map(clear_text, texts))


# In[137]:

coursera_texts = list(map(clear_text, coursera_texts))


# In[138]:

tmp_text = ""

for text in texts:
    tmp_text += text + " "

job_words = set(tmp_text.split(" "))


# In[139]:

tmp_text = ""

for text in coursera_texts:
    tmp_text += text + " "

coursera_words = set(tmp_text.split(" "))


# In[140]:

white_list = job_words & coursera_words


# In[141]:

tmp = []

for text in tqdm(texts):
    tmp_text = ""
    for word in text.split(" "):
        if word in white_list:
            tmp_text += word + " "

    tmp.append(tmp_text)

texts = tmp


# In[142]:

tmp = []

for text in tqdm(coursera_texts):
    tmp_text = ""
    for word in text.split(" "):
        if word in white_list:
            tmp_text += word + " "

    tmp.append(tmp_text)

coursera_texts = tmp


# In[143]:

all_texts = texts + coursera_texts


# In[144]:

vectorizer = text_a.CountVectorizer(input='content', stop_words='english', min_df=20)


# In[145]:

dtm = vectorizer.fit_transform(all_texts).toarray()


# In[146]:

dtm.shape


# In[147]:

vocab = np.array(vectorizer.get_feature_names())


# In[148]:

len(vocab)


# In[149]:

from sklearn import decomposition


# In[150]:

num_topics = 20 # num of jobs


# In[151]:

num_top_words = 20


# In[152]:

clf = decomposition.NMF(n_components=num_topics, random_state=1)


# In[153]:

doctopic = clf.fit_transform(dtm)


# In[154]:

topic_words = []


# In[155]:

for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])


# In[156]:

topic_names = [i[1] for i in topic_words]


# In[157]:

topics = [np.argmax(i) for i in doctopic]


# In[158]:

jobs_topic_names = pickle.load(open("job_topics.pck", "rb"))


# In[159]:

coursera_topic_names = [coursera[i]["title"] for i in coursera]


# In[160]:

len(jobs_topic_names+coursera_topic_names)


# In[161]:

from collections import defaultdict


# In[162]:

clusters = defaultdict(lambda: [])


# In[163]:

for a,b in zip(topics,[i + " (job)" for i  in jobs_topic_names] +[i + " (course)" for i in coursera_topic_names]):
    clusters[a].append((b, " (course)" in b))


# In[164]:

cluster_courses = []
cluster_jobs = []


# In[165]:

for a,b in enumerate(clusters):
    tmp_courses = []
    tmp_jobs = []
    
    if sum([i[1] for i in clusters[b]]) > 0:
        for k in clusters[b]:
            if k[1] == True:
                tmp_courses.append(k[0])
            else:
                tmp_jobs.append(k[0])
        cluster_courses.append(tmp_courses)
        cluster_jobs.append(tmp_jobs)


# In[166]:

api_dict = {}


# In[167]:

for job_list,courses_list  in zip(cluster_jobs,cluster_courses):
    for job in job_list:
        api_dict[job] = cluster_courses


# In[168]:

ALL_JOBS = api_dict.keys()


# In[169]:

ALL_JOBS = "{"
for job in api_dict.keys():
    ALL_JOBS+=job+","
ALL_JOBS = ALL_JOBS[:-1] + "}"


# In[170]:

from flask import Flask
app = Flask(__name__)

@app.route("/all_jobs")
def hello():
    return "{%s}" % ALL_JOBS

if __name__ == "__main__":
    app.run()


# In[128]:

import nltk


# In[ ]:



