
# coding: utf-8

# Step 1: Installing and importing required packages to import data from telegram messenger. 

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# In[2]:


import sys
get_ipython().system(u'{sys.executable} -m pip install telethon')


# In[3]:


from telethon import TelegramClient


# In[4]:


import pandas as pd
import numpy as np


# In[5]:


import re


# In[6]:


from collections import Counter


# In[7]:


import time


# Step 2: Authorising account in telegram to get an access to the data file. 

# In[8]:


api_id = 211270
api_hash = '4eeba3ebff5eebe405b657b2c019ff88'
phone_number = '+12064994265'


# In[9]:


client = TelegramClient('session_id', api_id=api_id, api_hash=api_hash)


# In[10]:


assert client.connect()


# In[19]:


if not client.is_user_authorized():
    client.send_code_request(phone_number)
    me = client.sign_in(phone_number, input('Enter code: '))
    print(me.stringify())

chat_id = 309823963


# Step 3: Loading data by batches of 10000 messages for safety. The code automatically filters every batch from unnecessary information. 

# In[20]:



def load_batch(chat_id, batch_size, last_batch_message_id):
    return client.get_message_history(chat_id, limit=batch_size, offset_id=last_batch_message_id)

def process_batch(batch):
    processed_messages = []
    for current_message in batch:
        try:
            current_message.media
            processed_messages.append(current_message)
        except:
            pass
    return processed_messages


# Step 4: Setting up a maximum amount of messages and summarizing the number of messages. 

# In[21]:


max_messages = 100000000

work_messages = []
time_elapsed = 0
batch_size = 10000
last_batch_message_id = -1

try:    
    while(len(work_messages) < max_messages):
        try:
            time_batch_start = time.time()
            loaded_messages = load_batch(
                chat_id, batch_size, last_batch_message_id)
            processed_messages = process_batch(loaded_messages)
            if len(processed_messages)==0:
                break
            work_messages.extend(processed_messages)
            last_batch_message_id = processed_messages[-1].id
            batch_duration = time.time() - time_batch_start
            time_elapsed += batch_duration
            print('Batch with {0} messages loaded for {1} seconds'.format(
                len(processed_messages), int(batch_duration)
            ))
            print('History loaded until {}'.format(work_messages[-1].date))
            print('Overall loaded {0} messages for {1} minutes\n'.format(
                len(work_messages), int(time_elapsed / 60)
            ))
        except RuntimeError:
            print('RUNTIME ERROR - try load batch again')
        
except KeyboardInterrupt:
    pass


# Step 5: Filtering data and creating my own data frame and dividing it by username. 

# In[22]:


corpus = pd.DataFrame(data = {
    'text': [mes.message for mes in work_messages],
    'is_media': [not mes.media == None for mes in work_messages],
    'is_bot': [mes.sender.bot for mes in work_messages],
    'writer_id': [mes.from_id for mes in work_messages],
    'username': [mes.sender.username for mes in work_messages],
    'first_name': [mes.sender.first_name for mes in work_messages],
    'last_name': [mes.sender.last_name for mes in work_messages],
    'mes_date': [mes.date for mes in work_messages],
    
})
corpus = corpus[~corpus.is_bot]
corpus = corpus[['mes_date', 'text', 'is_media', 'username', 'writer_id', 'first_name', 'last_name']]
corpus.text = corpus.text.fillna('').astype(str)

def remove_links_and_quotes(text):
    return ' '.join([word for word in text.split() if 'http' not in word and '@' not in word])
corpus.text = corpus.text.apply(lambda x: remove_links_and_quotes(x))

corpus = corpus[corpus.text.apply(lambda x: len(x) > 0)]
corpus.text = corpus.text.apply(lambda x: ' '.join(re.split('\W+', x.lower())))
corpus.sort_values('mes_date', inplace=True, ascending=True)

corpus.fillna('', inplace=True)
corpus['chatname'] = corpus['last_name'] + '_' + corpus['first_name'] + '(' + corpus['username'] + ')'

corpus.to_csv('joshua_chat.csv', index=None)

corpus.head(10)


# In[24]:


corpus = pd.read_csv('joshua_chat.csv')
corpus.fillna('', inplace=True)
corpus.mes_date = pd.to_datetime(corpus.mes_date)
corpus.head()


# In[25]:


import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def tokenize_me(file_text):
    #firstly let's apply nltk tokenization
    tokens = nltk.word_tokenize(file_text)

    #deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend(['this', 'like', 'so', 'and', 'hey', 'for', 'hi', 'that', 'on', 'in'])
    stop_words.extend(stopwords.words('english'))
    tokens = [i for i in tokens if ( i not in stop_words )]
    
    #cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return ' '.join(tokens)

corpus.text = corpus.text.apply(lambda x: tokenize_me(x))

corpus.to_csv('joshua_chat_tokens.csv', index=None)
corpus.head(10)


# Step 6: Identifying the number of messages by username.

# In[26]:


corpus.groupby('chatname')['text'].count().sort_values(ascending=False).head(15)


# In[27]:


corpus.groupby('chatname')['is_media'].sum().sort_values(ascending=False).astype(int).head(15)


# Step 7: Creating the graph of activity of our chat throughout the time.

# In[28]:


import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')

corpus.groupby(corpus.mes_date.dt.date).count().mes_date.plot(figsize=(24,5))
plt.title('Count of messages per day', fontsize=18)
plt.savefig('joshua_bydays.png')


# Step 8: Creating the graph of activity of number of messages based on time of the day. 

# In[29]:


import matplotlib.pylab as plt
get_ipython().magic(u'matplotlib inline')

corpus.groupby(corpus.mes_date.dt.hour).count().mes_date.plot(figsize=(24,5))
plt.title('Number of messages based on time of the day', fontsize=18)
plt.xticks(range(24))
plt.show()


# In[31]:


import sys
get_ipython().system(u'{sys.executable} -m pip install pymorphy2')


# Step 9: Filtering data by qulifiers and intensifiers. 

# In[32]:


import pymorphy2

users_text = corpus[corpus.chatname.map((corpus.groupby('chatname')['text'].count() > 50))]

users_text = users_text.groupby('chatname').agg(lambda x: ' '.join(x))['text']

morph = pymorphy2.MorphAnalyzer()


# In[33]:


# Apro
def normalise_string(input_str):
    bad_words = {'so','like','very','more',}
    norm_words_list = []
    for word in input_str.split():
        if not word.isdigit() and word not in bad_words:
            parsed = morph.parse(word)[0]
            if parsed.tag.POS not in {'NUMR', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ'}:
                norm_words_list.append(parsed.normal_form)
    return ' '.join(norm_words_list)

get_ipython().magic(u'time users_text_norm = users_text.apply(lambda x: normalise_string(x))')
users_text_norm.head()


# In[34]:


def filter_string(input_str):
    bad_words = {'so','like','very','more',}
    norm_words_list = []
    for word in input_str.split():
        if not word.isdigit() and word not in bad_words:
            parsed = morph.parse(word)[0]
            if parsed.tag.POS not in {'NUMR', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ'}:
                norm_words_list.append(word)
    return ' '.join(norm_words_list)

get_ipython().magic(u'time users_text_filter = users_text.apply(lambda x: filter_string(x))')
users_text_filter.head()


# In[36]:


import sys
get_ipython().system(u'{sys.executable} -m pip install gensim')


# Step 10: Removing meaningless and unnecessary words. 

# In[37]:


import gensim
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time

def get_words_by_threshold_report(text_series, min_df=0.02, max_df=0.75):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)

    counts = vectorizer.fit_transform(text_series)
    corpus_id2word = {value: key for key, value in vectorizer.vocabulary_.items()}

    print('Words in the library: {}\n'.format(counts.shape[1]))
    print('Words that are not applicable to the threshold: ', Counter(' '.join(text_series.values).split()).most_common(20), '\n')

    print('Words with the higher threshold:', [corpus_id2word[elem] for elem in np.argsort(np.array(counts.sum(axis=0)).ravel())[-40:][::-1]])
    print()
    print('Words with the lower threshold: ', [corpus_id2word[elem] for elem in np.argsort(np.array(counts.sum(axis=0)).ravel())[:20]])
    return counts, vectorizer, corpus_id2word

def get_lda_model(counts, corpus_id2word, num_topics=10, alpha_value=0.1, var_iter=50, passes=50):
    gensim_corpus = gensim.matutils.Scipy2Corpus(counts)
    t_start = time.time()
    lda = gensim.models.LdaModel(
        corpus=gensim_corpus,
        passes=passes,
        num_topics=num_topics,
        alpha=[alpha_value] * num_topics,
        iterations=var_iter,
        id2word=corpus_id2word,
        eval_every=0,
        random_state=42
    )
    current_score = lda.bound(gensim_corpus)
    time_ellapsed = int(time.time() - t_start)

    print('ELBO = {1:.4f}, time: {2} seconds'.format(
        alpha_value, current_score, time_ellapsed))
    
    gamma, _ = lda.inference(gensim_corpus)
    gamma = gamma / gamma.sum(1).reshape(-1, 1)
    
    return lda, gamma

def build_topic_words_report(lda, top_words_num=25, topic_names=None):
    if topic_names==None:
        topic_names = [''] * lda.num_topics
    topic_space = max(map(len, topic_names)) + 2
    
    if topic_space==2:
        for topic_index in range(lda.num_topics):
            print('{0} topic: {1}'.format(
                topic_index, ', '.join(list(zip(*lda.show_topic(topic_index, topn=top_words_num)))[0])))
        return
    
    for topic_index in range(lda.num_topics):
        print('{0} topic :{1: ^{2}}: {3}'.format(
            topic_index, topic_names[topic_index], topic_space,
            ', '.join(list(zip(*lda.show_topic(topic_index, topn=top_words_num)))[0])))
        
def build_topic_subjects_report(lda, gamma, text_series, topic_names=None):
    if topic_names==None:
        topic_names = [''] * lda.num_topics
    for topic_index in range(lda.num_topics):
        print('Topic {0} - {1}'.format(topic_index, topic_names[topic_index]))
        best_doc_indexes = gamma[:, topic_index].argsort()[-10:][::-1]
        
        subjects_with_proba = []
        for person_count, doc in enumerate(best_doc_indexes):
            if gamma[doc, topic_index] > 0.01 or person_count<3:
                subjects_with_proba.append('{0} - {1:.1%}'.format(text_series.index[doc], gamma[doc, topic_index]))
        print(', '.join(subjects_with_proba), '\n')


# Step 11: List of words in the initial form. 

# In[38]:


counts, vectorizer, corpus_id2word = get_words_by_threshold_report(users_text_norm, min_df=0.02, max_df=0.75)


# In[39]:


lda_norm, gamma_norm = get_lda_model(counts, corpus_id2word, num_topics=10)


# In[40]:


build_topic_words_report(lda_norm, top_words_num=20)


# In[41]:


build_topic_subjects_report(lda_norm, gamma_norm, users_text_norm)


# Step 12: Creating more stringent threshold. 

# In[42]:


day_text = corpus.groupby(corpus.mes_date.dt.date).agg(lambda x: ' '.join(x))['text']
print(day_text.shape)
day_text.head()


# In[43]:


morph = pymorphy2.MorphAnalyzer()
get_ipython().magic(u'time day_text_norm = day_text.apply(lambda x: normalise_string(x))')
day_text_norm.head()


# In[44]:


counts, vectorizer, corpus_id2word = get_words_by_threshold_report(day_text_norm, min_df=0.01, max_df=0.75)


# In[45]:


lda_day_norm, gamma_day_norm = get_lda_model(counts, corpus_id2word)


# In[46]:


build_topic_words_report(lda_day_norm, top_words_num=15)


# In[47]:


topic_dynamic = pd.DataFrame(day_text_norm)
topic_dynamic['topic'] = np.argmax(gamma_day_norm, axis=1)
topic_dynamic['message_count'] = day_text.apply(len)
topic_dynamic['date'] = topic_dynamic.index
topic_dynamic.sort_values('date', inplace=True)
topic_dynamic.head()


# In[48]:


import matplotlib as mpl
cmap = mpl.cm.hsv
normalize = mpl.colors.Normalize(vmin=1, vmax=10)

plt.figure(figsize=(30,10))
plt.plot(topic_dynamic.date, topic_dynamic.message_count, '-p')

for topic_index in range(10):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[:1], 
        x=topic_dynamic.date.values[:1], 
        color=cmap(normalize(topic_index)),
        label = '{0} - {1}'.format(topic_index + 1, ', '.join(list(zip(*lda_day_norm.show_topic(topic_index, topn=10)))[0]))
    )
for step in range(topic_dynamic.shape[0]):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[step: step+2], 
        x=topic_dynamic.date.values[step: step+2], 
        color=cmap(normalize(topic_dynamic.topic.iloc[step])),
    )
    
plt.legend(fontsize=12)
plt.ylim([0, topic_dynamic.message_count.max() * 1.5])
plt.ylabel('Number of messages per week')
plt.title('Dynamic of changes of topic by day', fontsize=21)
plt.savefig('joshua_day_dynamic_norm.png')
plt.show()


# In[49]:



counts, vectorizer, corpus_id2word = get_words_by_threshold_report(day_text_norm, min_df=0.01, max_df=0.45)


# In[50]:


lda_day_norm, gamma_day_norm = get_lda_model(counts, corpus_id2word)


# In[51]:


build_topic_words_report(lda_day_norm, top_words_num=15)


# In[52]:


topic_dynamic = pd.DataFrame(day_text_norm)
topic_dynamic['topic'] = np.argmax(gamma_day_norm, axis=1)
topic_dynamic['message_count'] = day_text.apply(len)
topic_dynamic['date'] = topic_dynamic.index
topic_dynamic.sort_values('date', inplace=True)
topic_dynamic.head()


# In[53]:


import matplotlib as mpl
cmap = mpl.cm.hsv
normalize = mpl.colors.Normalize(vmin=1, vmax=10)

plt.figure(figsize=(30,20))
plt.plot(topic_dynamic.date, topic_dynamic.message_count, '-p')

for topic_index in range(10):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[:1], 
        x=topic_dynamic.date.values[:1], 
        color=cmap(normalize(topic_index)),
        label = '{0} - {1}'.format(topic_index + 1, ', '.join(list(zip(*lda_day_norm.show_topic(topic_index, topn=10)))[0]))
    )
for step in range(topic_dynamic.shape[0]):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[step: step+2], 
        x=topic_dynamic.date.values[step: step+2], 
        color=cmap(normalize(topic_dynamic.topic.iloc[step])),
    )
    
plt.legend(fontsize=12)
plt.ylim([0, topic_dynamic.message_count.max() * 1.5])
plt.ylabel('Number of messages per week')
plt.title('Dynamic of changes of topics by day', fontsize=21)
plt.savefig('joshua_day_dynamic_norm_2.png')
plt.show()


# In[54]:



week_text = corpus.groupby(corpus.mes_date.apply(lambda x: '{1}-{0}'.format(x.week, x.year))).agg(lambda x: ' '.join(x))['text']
print(week_text.shape)
week_text.head()


# In[55]:


morph = pymorphy2.MorphAnalyzer()
get_ipython().magic(u'time week_text_norm = week_text.apply(lambda x: normalise_string(x))')
week_text_norm.head()


# In[56]:



counts, vectorizer, corpus_id2word = get_words_by_threshold_report(week_text_norm, min_df=0.1, max_df=0.5)


# In[57]:


lda_week_norm, gamma_week_norm = get_lda_model(counts, corpus_id2word)


# In[58]:


build_topic_words_report(lda_week_norm, top_words_num=15)


# In[59]:


def weekpair_2date(pair):
    atime = time.strptime('{} {} 1'.format(*pair.split('-')), '%Y %W %w')
    return pd.to_datetime('{0}-{1}-{2}'.format(atime.tm_year, atime.tm_mon, atime.tm_mday))

topic_dynamic = pd.DataFrame(week_text_norm)
topic_dynamic['topic'] = np.argmax(gamma_week_norm, axis=1)
topic_dynamic['message_count'] = week_text.apply(len)
topic_dynamic['date'] = list(map(weekpair_2date, topic_dynamic.index))
topic_dynamic.sort_values('date', inplace=True)
topic_dynamic.head()


# In[60]:



import matplotlib as mpl
cmap = mpl.cm.hsv
normalize = mpl.colors.Normalize(vmin=1, vmax=10)

plt.figure(figsize=(30,10))
plt.plot(topic_dynamic.date, topic_dynamic.message_count, '-p')

for topic_index in range(10):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[:1], 
        x=topic_dynamic.date.values[:1], 
        color=cmap(normalize(topic_index)),
        label = '{0} - {1}'.format(topic_index + 1, ', '.join(list(zip(*lda_week_norm.show_topic(topic_index, topn=10)))[0]))
    )
for step in range(topic_dynamic.shape[0]):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[step: step+2], 
        x=topic_dynamic.date.values[step: step+2], 
        color=cmap(normalize(topic_dynamic.topic.iloc[step])),
    )
    
plt.legend(fontsize=12)
plt.ylim([0, topic_dynamic.message_count.max() * 1.5])
plt.ylabel('Number of messages per week')
plt.title('Dynamic of changes of topics by day', fontsize=21)
plt.savefig('joshua_week.png')
plt.show()


# Last step: Same model in normalized form. 

# In[61]:


counts, vectorizer, corpus_id2word = get_words_by_threshold_report(week_text_norm, min_df=0.1, max_df=0.75)


# In[62]:


lda_week_norm, gamma_week_norm = get_lda_model(counts, corpus_id2word)


# In[63]:


build_topic_words_report(lda_week_norm, top_words_num=15)


# In[64]:


def weekpair_2date(pair):
    atime = time.strptime('{} {} 1'.format(*pair.split('-')), '%Y %W %w')
    return pd.to_datetime('{0}-{1}-{2}'.format(atime.tm_year, atime.tm_mon, atime.tm_mday))

topic_dynamic = pd.DataFrame(week_text_norm)
topic_dynamic['topic'] = np.argmax(gamma_week_norm, axis=1)
topic_dynamic['message_count'] = week_text.apply(len)
topic_dynamic['date'] = list(map(weekpair_2date, topic_dynamic.index))
topic_dynamic.sort_values('date', inplace=True)
topic_dynamic.head()


# In[65]:


import matplotlib as mpl
cmap = mpl.cm.hsv
normalize = mpl.colors.Normalize(vmin=1, vmax=10)

plt.figure(figsize=(30,10))
plt.plot(topic_dynamic.date, topic_dynamic.message_count, '-p')

for topic_index in range(10):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[:1], 
        x=topic_dynamic.date.values[:1], 
        color=cmap(normalize(topic_index)),
        label = '{0} - {1}'.format(topic_index + 1, ', '.join(list(zip(*lda_week_norm.show_topic(topic_index, topn=10)))[0]))
    )
for step in range(topic_dynamic.shape[0]):
    plt.fill_between(
        y1=0, y2=topic_dynamic.message_count[step: step+2], 
        x=topic_dynamic.date.values[step: step+2], 
        color=cmap(normalize(topic_dynamic.topic.iloc[step])),
    )
    
plt.legend(fontsize=12)
plt.ylim([0, topic_dynamic.message_count.max() * 1.5])
plt.ylabel('Number of messages per week')
plt.title('Dynamic of changes of topics by day', fontsize=21)
plt.savefig('joshua_week_2.png')
plt.show()

