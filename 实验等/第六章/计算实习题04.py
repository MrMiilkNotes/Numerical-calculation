#!/usr/bin/env python
# coding: utf-8

# # 计算实习题四

# > 本章的算法较为简单，推导过程较为繁琐，因此就不再自己推导。在课本P207给出了详细流程

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#计算实习题四" data-toc-modified-id="计算实习题四-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>计算实习题四</a></span><ul class="toc-item"><li><span><a href="#实现算法" data-toc-modified-id="实现算法-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>实现算法</a></span></li><li><span><a href="#测试" data-toc-modified-id="测试-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>测试</a></span></li></ul></li></ul></div>

# In[1]:


# 必要库文件
import numpy as np
import scipy


# ## 实现算法

# - 输入`A`，`b`构造计算方法，然后调用唯一接口`cacu()`进行计算，`x`默认从`[0,0,...0]`开始迭代

# In[2]:


class CG:
    """共轭梯度法求解线性方程Ax=b"""
    def __init__(self, A, b, ):
        self.n = b.shape[0]
        self.b = b
        self.A = A
        self.p = None
        self.alpha = 0
        self.r = None
        self.k = 0
        
    def cacu(self, x = None):
        """具体迭代过程"""
        if x == None :
            x = np.zeros((self.n, 1))
        k = self.n
        self.r = self.p = b - np.dot(A, x)
        while self.__cacu_alpha() and k > 0 :    # 迭代n次或者达到终止条件
            k -= 1
            x = x + self.alpha * self.p
            r = self.r - self.alpha * np.dot(A, self.p)
            B = self.__get_beta(r)
            self.p = self.r + B * self.p
        self.k = self.n - k
        return x
        
    def __cacu_alpha(self, ):
        """计算Alpha，同时检验算法是否可以继续"""
        if np.all(self.r == 0):
            return False
        tmp1 = np.sum(self.r * self.r)
        tmp2 = np.sum(self.p * np.dot(A, self.p))
        if(tmp2 == 0):
            return False
        self.alpha = tmp1 / tmp2
        return True
    
    def __get_beta(self, r):
        """计算Beta"""
        tmp = np.sum(r*r) / np.sum(self.r * self.r)
        self.r = r
        return tmp
    
    def times(self):
        return self.k


# ## 测试

# - 使用课本P207例题进行测试
# - 结果一致

# In[3]:


A = np.array([
    [3., 1],
    [1, 2]
])
b = np.array([
    [5],
    [5]
])


# In[4]:


# 构造求解器
cg_solu = CG(A, b)


# In[5]:


# 解出答案
cg_solu.cacu()


# In[6]:


print("计算次数：", cg_solu.times())


# - 输入数据量太小，这里构造一个`100 * 100`的三对角矩阵`A`作为测试

# In[7]:


# 生成数据
A = np.zeros((100, 100))
for i in range(100): #generate A
    for j in range(100):
        if (i == j):
            A[i, j] = 2
        if (abs(i - j) == 1):
            A[i, j] = A[j, i] = -1
b = np.ones((100, 1))  #generate b


# - 利用`numpy`库的求逆，真实解为：

# In[8]:


np.dot(np.linalg.inv(A), b).T


# - 使用我们的方法进行求解
# > 可以看到的确得到了正确解，计算次数小于`n=100`，只用了50次迭代

# In[9]:


cg_solu = CG(A, b)
print(cg_solu.cacu().T)


# In[10]:


print("计算次数：", cg_solu.times())


# In[ ]:




