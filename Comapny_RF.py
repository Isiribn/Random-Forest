#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv("Company_Data.csv")
data.head()


# In[3]:


data.tail(10)


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.isnull().any()


# In[7]:


data.info()


# In[11]:


x=data['Sales']


# In[110]:


import matplotlib.pyplot as plt
plt.hist(x)


# In[24]:


bin=[0,5,10,15,20]
sales=pd.cut(x,bin,labels=['1','2','3','4'])


# In[25]:


sales.head()


# In[26]:


sales.isnull().any()


# In[27]:


sales[sales.isnull()]


# In[28]:


data.iloc[174]


# In[29]:


sales_df=pd.DataFrame(sales)


# In[30]:


sales_df


# In[42]:


sales_df=sales_df.rename({'Sales':'Sale_Types'},axis=1)
sales_df


# In[56]:


data_df=pd.concat([data,sales_df],axis=1)
data_df


# In[57]:


data_df.isnull().any()


# In[58]:


data_df['Sale_Types']=data_df['Sale_Types'].fillna('1')


# In[59]:


data_df.iloc[174]


# In[60]:


data_df


# In[61]:


data_df.isnull().any()


# In[62]:


string_col=['ShelveLoc','Urban','US']
from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
for i in string_col:
    data_df[i]=num.fit_transform(data_df[i])


# In[63]:


data_df


# In[64]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data_df,test_size=0.3)


# In[65]:


trainx=train.iloc[:,1:11]
trainy=train.iloc[:,11]
testx=test.iloc[:,1:11]
testy=test.iloc[:,11]


# In[66]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[ ]:



Diabetes['rf_pred'] = rf.predict(X)
cols = ['rf_pred',' Class variable']
Diabetes[cols].head()
Diabetes[" Class variable"]


from sklearn.metrics import confusion_matrix
confusion_matrix(Diabetes[' Class variable'],Diabetes['rf_pred']) # Confusion matrix

pd.crosstab(Diabetes[' Class variable'],Diabetes['rf_pred'])



print("Accuracy",(497+268)/(497+268+0+3)*100)

# Accuracy is 99.609375
Diabetes["rf_pred"]


# In[67]:


rf.fit(trainx,trainy)


# In[68]:


rf.estimators_


# In[69]:


rf.classes_


# In[70]:


rf.n_classes_


# In[71]:


rf.n_features_


# In[73]:


rf.n_outputs_


# In[74]:


rf.oob_score_


# In[75]:


rf.predict(testx)


# In[76]:


testy


# In[77]:


from sklearn.metrics import confusion_matrix
confusion_matrix(testy,rf.predict(testx))


# In[78]:


pd.crosstab(testy,rf.predict(testx))


# In[80]:


from sklearn import metrics
print("Test Accuracy=",metrics.accuracy_score(testy,rf.predict(testx)))


# In[81]:


print("Train Accuracy=", metrics.accuracy_score(trainy,rf.predict(trainx)))


# # Improving accuracy after Standardize

# In[86]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()


# In[94]:


data_df.info()


# In[96]:


data_df.drop("Sales",inplace=True,axis=1)


# In[97]:


data_df


# In[155]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data_df,test_size=0.35)


# In[156]:


trainx=train.iloc[:,0:10]
trainy=train.iloc[:,10]
testx=test.iloc[:,0:10]
testy=test.iloc[:,10]


# In[157]:


trainx=scale.fit_transform(trainx)


# In[158]:


testx=scale.fit_transform(testx)


# In[169]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")


# In[170]:


model.fit(trainx,trainy)


# In[171]:


model.predict(testx)


# In[172]:


testy


# In[173]:


from sklearn import metrics
print("Train accuracy=",metrics.accuracy_score(trainy,model.predict(trainx)))


# In[174]:


print("Test accuracy=",metrics.accuracy_score(testy,model.predict(testx)))


# In[177]:


#Visualizing
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(estimator, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('RandomForest.png')
Image(graph.create_png())


# In[175]:


estimator=model.estimators_[5]


# In[176]:


estimator


# In[ ]:




