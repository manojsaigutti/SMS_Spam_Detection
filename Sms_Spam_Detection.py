#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[18]:


import chardet
with open('spam.csv', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']
sms = pd.read_csv('spam.csv', encoding=encoding )


# In[19]:


sms.head()


# In[27]:


sms.shape


# In[28]:


sms.drop_duplicates(inplace=True)


# In[30]:


sms.reset_index(drop=True, inplace=True)


# In[31]:


sms.shape


# In[33]:


sms['v1'].value_counts()


# In[34]:


plt.figure(figsize=(8,5))
sns.countplot(x='v1', data=sms)
plt.xlabel('SMS Classification')
plt.ylabel('Count')
plt.show()


# In[44]:


corpus = []
ps = PorterStemmer()

for i in range(0, sms.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.iloc[i, 0])  # Accessing the first column by index
    message = message.lower()  # Converting the entire message into lower case
    words = message.split()  # Tokenizing the review by words
    words = [word for word in words if word not in set(stopwords.words('english'))]  # Removing the stop words
    words = [ps.stem(word) for word in words]  # Stemming the words
    message = ' '.join(words)  # Joining the stemmed words
    corpus.append(message)  # Building a corpus of messages.


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


# In[47]:


y = pd.get_dummies(sms['v1'])
y = y.iloc[:, 1].values


# In[48]:


y


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[51]:


best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.0,1.1,0.1):
    temp_classifier = MultinomialNB(alpha=i)
    temp_classifier.fit(X_train, y_train)
    temp_y_pred = temp_classifier.predict(X_test)
    score = accuracy_score(y_test, temp_y_pred)
    print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
    if score>best_accuracy:
        best_accuracy = score
        alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[52]:


classifier = MultinomialNB(alpha=0.0)
classifier.fit(X_train, y_train)


# In[53]:


y_pred = classifier.predict(X_test)


# In[54]:


y_pred


# In[55]:


acc_s = accuracy_score(y_test, y_pred)*100


# In[56]:


print("Accuracy Score {} %".format(round(acc_s,2)))


# In[57]:


def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)


# In[58]:


result = ['Wait a minute, this is a SPAM!','Ohhh, this is a normal message.']


# In[59]:


msg = "Hi! You are pre-qulified for Premium SBI Credit Card. Also get Rs.500 worth Amazon Gift Card*, 10X Rewards Point* & more. Click "

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[60]:


msg = "[Update] Congratulations Nile Yogesh, You account is activated for investment in Stocks. Click to invest now: "

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[61]:


msg = "Your Stock broker FALANA BROKING LIMITED reported your fund balance Rs.1500.5 & securities balance 0.0 as on end of MAY-20 . Balances do not cover your bank, DP & PMS balance with broking entity. Check details at YOGESHNILE.WORK4U@GMAIL.COM. If email Id not correct, kindly update with your broker."

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[ ]:




