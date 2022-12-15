#!/usr/bin/env python
# coding: utf-8

# ### KNN from Scratch

# In[1]:


import numpy as np
from collections import Counter 


# In[2]:


def euclidean_distance (x1, x2):
    return np.sqrt (np.sum((x1-x2)**2))

# Define a class

class KNN:
    # Default value for k is 3
    
    def __init__(self, k=3):
        self.k = k
    
    # Training process X->Training samples, y->Training labels
    # Store training sample and training labels 
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    # Predicting process
    # X represents many samples
    def predict (self, X):
        predicted_labels = [self._predict(x) for x in X]
        # convert the list to numpy array  
        return np.array(predicted_labels)
    
    def _predict (self, x):
        # compute distances 
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]
        
        # get k nearest samples, labels 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # majority vote, most common class label 
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
        


# In[ ]:




