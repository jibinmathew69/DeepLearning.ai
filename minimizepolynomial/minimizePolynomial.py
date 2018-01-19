
# coding: utf-8

# In[107]:


import tensorflow as tf
import numpy as np


# In[108]:


coefficient = np.array([[1],[-34],[25]])


# In[109]:


w = tf.Variable([0],dtype=tf.float32)


# In[110]:


x = tf.placeholder(tf.float32,[3,1])


# In[111]:


cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]


# In[112]:


train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# In[113]:


init = tf.global_variables_initializer()


# In[114]:


with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train,feed_dict={x:coefficient})
    print(session.run(w))

