
# coding: utf-8

# In[6]:


import time
from contextlib import contextmanager


# In[7]:


#how to use with
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def testtimer():
    with timer("hi"):
        print("done")       


# In[10]:


#hide in modules
if __name__=="__main__":
    testtimer()


# In[ ]:


#make only 1 thread to run
import os; 
os.environ['OMP_NUM_THREADS'] = '1'

