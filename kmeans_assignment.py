#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[45]:


df= pd.read_csv('Mall_Customers.csv',index_col='CustomerID')


# In[46]:


df.head()


# In[47]:


df.info()


# In[48]:


df.describe()


# In[49]:


df.isnull()


# In[50]:


df.drop_duplicates(inplace=True)


# In[51]:


x=df.iloc[:,[2,3]].values
print(x)


# In[52]:


df.isnull().sum()


# In[53]:


wcss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[54]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,color='red')
plt.title('elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()


# In[55]:


kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_pred=kmeans.fit_predict(x)
plt.figure(figsize=(10,6))
for i in range(5):
    plt.scatter(x[y_pred==i,0],x[y_pred==i,1])
    


# In[ ]:




