#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("Fraud_check.csv")
data.head()


# In[2]:


data.info()


# In[5]:


data.isnull().any()


# In[7]:


import numpy as np
x=np.where(data['Taxable.Income']<=30000,'risky','good')
x


# In[8]:


x=pd.DataFrame(x)
x


# In[12]:


x=x.rename({0:'TIncome'},axis=1)
x


# In[13]:


data_df=pd.concat([data,x],axis=1)
data_df


# In[14]:


data_df.drop('Taxable.Income',inplace=True,axis=1)


# In[15]:


data_df


# In[19]:


string_col=["Undergrad",'Marital.Status','Urban','TIncome']
from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
for i in string_col:
 data_df[i]=num.fit_transform(data_df[i])


# In[20]:


data_df


# In[21]:


data_df.isnull().any()


# In[22]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data_df,test_size=0.3)


# In[23]:


trainx=train.iloc[:,0:5]
trainy=train.iloc[:,5]
testx=test.iloc[:,0:5]
testy=test.iloc[:,5]


# In[26]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[27]:


rf.fit(trainx,trainy)


# In[28]:


rf.predict(testx)


# In[29]:


testy


# In[31]:


from sklearn import metrics
print("Train accuracy=",metrics.accuracy_score(trainy,rf.predict(trainx)))


# In[32]:


print("Test accuracy=",metrics.accuracy_score(testy,rf.predict(testx)))


# In[35]:


estimator=rf.estimators_[3]


# In[36]:


estimator


# In[39]:


#Visualizing
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(estimator, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('RandomForest_fraud.png')
Image(graph.create_png())


# In[ ]:





# In[ ]:


#Feature selection-dropping City.Population


# In[40]:


data_df.columns


# In[41]:


data_df.drop("City.Population",inplace=True, axis=1)


# In[42]:


data_df


# In[43]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data_df,test_size=0.3)


# In[44]:


trainx=train.iloc[:,0:4]
trainy=train.iloc[:,4]
testx=test.iloc[:,0:4]
testy=test.iloc[:,4]


# In[45]:


from sklearn.ensemble import RandomForestClassifier
mod= RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[46]:


mod.fit(trainx,trainy)


# In[47]:


mod.predict(testx)


# In[48]:


testy


# In[49]:


from sklearn import metrics
print("Train accuracy=",metrics.accuracy_score(trainy,mod.predict(trainx)))


# In[50]:


print("Test accuracy=",metrics.accuracy_score(testy,mod.predict(testx)))


# In[ ]:




