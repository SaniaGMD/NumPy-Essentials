#!/usr/bin/env python
# coding: utf-8

# # <center>NumPy Essentials: A Comprehensive Guide to NumPy<center>

# NumPy, short for **Numerical Python**, is a Python library that adds support for large, multi-dimensional arrays and matrices, as well as a large collection of high-level mathematical functions to operate on these arrays. NumPy is one of the foundational libraries used in data science and machine learning, as it provides a powerful toolset for **data manipulation, analysis, and computation.**
# 
# Arrays in NumPy are similar to lists in Python but are more efficient and powerful for numerical operations. Here are some key features of NumPy arrays:
# 
# - **Homogeneous**: All elements in a NumPy array must have the same data type, which is specified when the array is created.
# - **Fixed-size**: The size of a NumPy array is fixed when it is created and cannot be changed.
# - **Multi-dimensional**: NumPy arrays can have any number of dimensions, from one to many.

# ## Applications of NumPy

# - Mathematics (MATLAB replacement)
# - Plotting (Matplotlib)
# - Backend (Pandas, Connect 4, Digital Photography, Image processing)
# - Machine Learning

# ## Getting Started with NumPy

# To use NumPy in your Python code, you first need to install it. You can do this using pip, the Python package manager, by running the following command:

# In[56]:


pip install numpy


# This statement creates an **alias “np”** for the NumPy library, which makes it easier to use in your code.
# 
# Let’s look at some examples of how to use NumPy for data science tasks.

# ## Creating & Inspecting NumPy Array

# You can create NumPy arrays using a variety of functions provided by the library. Here are a few examples:

# In[57]:


#importing numpy
import numpy as np 

a = np.array([1,2,3])
print(a)


# In[58]:


b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)


# In[59]:


# Get Dimension of a array
print(a.ndim)
print(b.ndim)


# In[60]:


# Get Shape
print(a.shape)
print(b.shape)


# In[61]:


# Get Type
a.dtype


# In[62]:


# Get Size
print(a.itemsize)


# In[63]:


# Get total size
print(a.nbytes)


# In[64]:


# Get the lenght of array
len(a)


# In[65]:


# Get the name of the datatype
a.dtype.name


# ## Initial Placeholders

# 1. Create a 3x5 array filled with zeros.
# 2. Create a 2x2 array filled with ones.
# 3. Create an array of values from 10 to 25 (excluding 25) with a step of 5.
# 4. Create an array of 9 evenly spaced values between 0 and 2.
# 5. Create a 2x2 array filled with the value 7.
# 6. Create a 2x2 identity matrix.
# 7. Create a 2x2 array with random values between 0 and 1.
# 8. Create a 3x2 array without initializing its entries.

# In[66]:


np.zeros((3,5))


# In[67]:


np.ones((2,2))


# In[68]:


np.arange(10,25,5)


# In[69]:


np.linspace(0,2,9)


# In[70]:


np.full((2,2),7)


# In[71]:


np.eye(2)


# In[72]:


np.random.random((2,2))


# In[73]:


np.empty((3,2))


# ## Data Types

# - np.int32:            Signed 32-bit integer type 
# - np.uint64:            Unsigned 64-bit integer type 
# - np.float64:           Double-precision floating point 
# - np.complex128:        Complex number represented by two 64-bit floats 
# - np.bool:              Boolean type storing True and False values 
# - np.object:            Python object type 
# - np.string_:           Fixed-length string type 
# - np.unicode_:          Fixed-length unicode type 

# ## Subsetting, Slicing & Indexing

# In[74]:


a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a) 


# In[75]:


# Get a specific element [r, c]
a[1, 5]


# In[76]:


# Get a specific row 
a[0, :]


# In[77]:


# Getting a little more fancy [startindex:endindex:stepsize]
a[0, 1:-1:2]


# In[78]:


#updating an elemn
a[1,5] = 20
a[:,2] = [1,2]
print(a)


# ## Array Mathematics

# In[79]:


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])


# In[80]:


np.add(a, b)            # Output: array([5, 7, 9])
np.subtract(a, b)       # Output: array([-3, -3, -3])
np.multiply(a, b)       # Output: array([4, 10, 18])
np.divide(a, b)         # Output: array([0.25, 0.4 , 0.5 ])
np.power(a, b)          # Output: array([  1,  32, 729])
np.sqrt(a)              # Output: array([1.        , 1.41421356, 1.73205081])
np.exp(a)            # Output: array([ 2.71828183,  7.3890561 , 20.08553692])
np.sin(a)            # Output: array([0.84147098, 0.90929743, 0.14112001])
np.cos(a)            # Output: array([ 0.54030231, -0.41614684, -0.9899925 ])
np.log(a)            # Output: array([0.        , 0.69314718, 1.09861229])

#Note: you are also allowed to use mathematical symbols directly like +, -, * etc.


#  ## Linear Algebra

# In[81]:


a = np.ones((2,3))
print(a)

b = np.full((3,2), 2)
print(b)

# For Matrix Multiplction
np.matmul(a,b)


# In[82]:


# Find the determinant
c = np.identity(3)
np.linalg.det(c)   #output: 1


# **Reference docs for Linear Algebra operations:**
# 
# https://docs.scipy.org/doc/numpy/reference/routines.linalg.html
# 
# - Determinant
# - Trace
# - Singular Vector Decomposition
# - Eigenvalues
# - Matrix Norm
# - Inverse etc.

# ## Statistics
np.mean(arr)           Compute the arithmetic mean along a specified axis.
np.median(arr)         Compute the median along a specified axis.
np.std(arr)            Compute the standard deviation along a specified axis.
np.var(arr)            Compute the variance along a specified axis.
np.max(arr)            Find the maximum value along a specified axis.
np.min(arr)            Find the minimum value along a specified axis.
np.sum(arr)            Compute the sum of all elements along a specified axis.
np.prod(arr)           Compute the product of all elements along a specified axis.
np.percentile(arr, q)  Compute the q-th percentile of the data along a specified axis.
np.corrcoef(arr)       Compute the correlation matrix of a 2D array.
np.cov(arr)            Estimate a covariance matrix of a 2D array.
# ## Reorganizing Arrays

# In[83]:


before = np.array([[1,2,3,4],[5,6,7,8]])
print(before)

after = before.reshape((4, 2))
print(after)       


# In[84]:


# Vertically stacking vectors
v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8])

np.vstack([v1,v2,v1,v2])


# In[85]:


# Horizontal  stack
h1 = np.ones((2,4))
h2 = np.zeros((2,2))

np.hstack((h1,h2))


# ## Array Manipulation
# To take the transpose of an array
np.transpose(a)# Adding/removing elements
np.append(h,g)
np.insert(a, 1, 5)
np.delete(a, [1])# Combining arrays
np.concatenate(a,b)# Sorting arrays
a.sort()# Copying arrays
np.copy(a)
# ## Important Note

# I’m working on **#100daysofMLCode**, where I am starting with the beginner level, I’ll post the weekly content or more in a week in a form of a crux. If you’re a beginner and want to enhance your knowledge of Data science along with practice implementation, we can get connected through LinkedIn, Medium, and GitHub here.

# # LinkedIn

# Here is my LinkedIn profile in case you want to connect with me. I’ll be happy to be connected with you. <br>
# Lets Connected: https://www.linkedin.com/in/sania-mohiu-ud-din-277047193 <br>
# GitHub: https://github.com/SaniaGMD
