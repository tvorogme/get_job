
# coding: utf-8

# In[1]:

import numpy as np
import pickle
import sklearn.feature_extraction.text as text_a
import re
from collections import Counter
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm


# In[2]:

lemmatizer = WordNetLemmatizer()


# In[3]:

texts = pickle.load(open("answer.pck", "rb"))


# In[4]:

coursera = pickle.load(open("coursera.p", "rb"))


# In[5]:

coursera_texts = [coursera[i]["description"] for i in coursera]


# In[6]:

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


# In[7]:

nltk.data.path.append("/home")


# In[8]:

texts = list(map(clear_text, texts))


# In[9]:

coursera_texts = list(map(clear_text, coursera_texts))


# In[10]:

tmp_text = ""

for text in texts:
    tmp_text += text + " "

job_words = set(tmp_text.split(" "))


# In[11]:

tmp_text = ""

for text in coursera_texts:
    tmp_text += text + " "

coursera_words = set(tmp_text.split(" "))


# In[12]:

white_list = job_words & coursera_words


# In[13]:

tmp = []

for text in tqdm(texts):
    tmp_text = ""
    for word in text.split(" "):
        if word in white_list:
            tmp_text += word + " "

    tmp.append(tmp_text)

texts = tmp


# In[14]:

tmp = []

for text in tqdm(coursera_texts):
    tmp_text = ""
    for word in text.split(" "):
        if word in white_list:
            tmp_text += word + " "

    tmp.append(tmp_text)

coursera_texts = tmp


# In[15]:

all_texts = texts + coursera_texts


# In[16]:

vectorizer = text_a.CountVectorizer(input='content', stop_words='english', min_df=20)


# In[17]:

dtm = vectorizer.fit_transform(all_texts).toarray()


# In[18]:

dtm.shape


# In[21]:

dtm = dtm.astype('float32')


# In[22]:

vocab = np.array(vectorizer.get_feature_names())


# In[1]:

len(vocab)


# In[6]:

import numpy as np
np.linalg.svd(np.random.randn(3000,2000))


# In[24]:

from sklearn import decomposition


# In[25]:

num_topics = 20 # num of jobs


# In[26]:

num_top_words = 20


# In[27]:

clf = decomposition.PCA(n_components=num_topics)


# In[ ]:

doctopic = clf.fit_transform(dtm)


# In[1]:

topic_words = []


# In[2]:

for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])


# In[ ]:

topic_names = [i[1] for i in topic_words]


# In[ ]:

topics = [np.argmax(i) for i in doctopic]


# In[ ]:

jobs_topic_names = pickle.load(open("job_topics.pck", "rb"))


# In[ ]:

coursera_topic_names = [coursera[i]["title"] for i in coursera]


# In[ ]:

len(jobs_topic_names+coursera_topic_names)


# In[ ]:

from collections import defaultdict


# In[ ]:

clusters = defaultdict(lambda: [])


# In[ ]:

for a,b in zip(topics,[i + " (job)" for i  in jobs_topic_names] +[i + " (course)" for i in coursera_topic_names]):
    clusters[a].append((b, " (course)" in b))


# In[ ]:

cluster_courses = []
cluster_jobs = []


# In[ ]:

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


# In[ ]:

api_dict = {}


# In[ ]:

for job_list,courses_list  in zip(cluster_jobs,cluster_courses):
    for job in job_list:
        api_dict[job] = cluster_courses


# In[ ]:

ALL_JOBS = api_dict.keys()


# In[ ]:

ALL_JOBS = "{"
for job in api_dict.keys():
    ALL_JOBS+=job+","
ALL_JOBS = ALL_JOBS[:-1] + "}"


# In[ ]:

from flask import Flask
app = Flask(__name__)

@app.route("/all_jobs")
def hello():
    return "{%s}" % ALL_JOBS

if __name__ == "__main__":
    app.run()


# In[ ]:



