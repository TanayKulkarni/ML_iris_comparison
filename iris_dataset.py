
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris


# In[4]:


iris = load_iris()


# In[6]:


x = iris.data
y = iris.target


# In[9]:


x.shape


# In[12]:


y.shape


# In[15]:


y


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[29]:


x_train


# In[21]:


y_train.shape


# In[24]:


from sklearn import tree


# In[32]:


clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
clf.predict([[2.3,5.6,8.7,4.0]])


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)


# In[62]:


from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(x_train,y_train)
pred1 = clf1.predict(x_test)
clf1.predict([[2.0,5.6,0.7,0]])


# In[66]:


accuracy_score(y_test,pred1)


# In[69]:


from sklearn.svm import SVC
clf2 = SVC()
clf2.fit(x_train,y_train)
pred2 = clf2.predict(x_test)


# In[72]:


accuracy_score(y_test,pred2)


# In[77]:


from sklearn.ensemble import RandomForestClassifier
                           
clf3 = RandomForestClassifier(max_depth=2, random_state=0)
clf3.fit(x_train,y_train)
pred3 = clf3.predict(x_test)


# In[79]:


accuracy_score(y_test,pred3)

