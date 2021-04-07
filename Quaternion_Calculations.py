#!/usr/bin/env python
# coding: utf-8

# In[16]:


import transforms3d.quaternions as tf
import sympy as sp
import numpy as np
sympy.init_printing()


# In[17]:


#symbols for calculations
a0 = sp.symbols("a0")
a1 = sp.symbols("a1")
a2 = sp.symbols("a2")
a3 = sp.symbols("a3")

b0 = sp.symbols("b0")
b1 = sp.symbols("b1")
b2 = sp.symbols("b2")
b3 = sp.symbols("b3")



i = sp.symbols("i")
j = sp.symbols("j")
k = sp.symbols("k")


################################

################################


# In[18]:


#Quaternion multiplication
#def qmult(q0, q1):
#    res = sp.Matrix(tf.qmult(q0, q1))
#    res.simplify()
#    return res

#Congugate of a quaternions number
#def qconj(q):
#    return sp.Matrix([q[0], -q[1], -q[2], -q[3]])

#Norm of a quaternion number
#def qnorm(q):
#    return sp.sqrt(qmult(q, qconj(q))[0])


# In[19]:


U = np.array([a0,a1,a2,a3]) # A quaternion number
W = np.array([b0,b1,b2,b3])

matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
matrix2 = np.array([[0,22,3,43],[15,61,71,8],[19,0,11,112],[0,114,2,0]])

matrix


# In[20]:


G = np.array([2,-1,3,1])
H = np.array([3,2,-1,-1])


# In[33]:


#puts quaternion into its proper form
def qfancy(array):
    return array[0] + array[1]*i + array[2]*j + array[3]*k
    
#multiplies two quaternions giving us a new array    
def qmult(array1,array2):
    return np.array([array1[0]*array2[0] - array1[1]*array2[1] - array1[2]*array2[2] - array1[3]*array2[3],
                     array1[0]*array2[1] + array1[1]*array2[0] + array1[2]*array2[3] - array1[3]*array2[2],
                     array1[0]*array2[2] + array1[2]*array2[0] - array1[1]*array2[3] + array1[3]*array2[1],               
                     array1[0]*array2[3] + array1[3]*array2[0] + array1[1]*array2[2] - array1[2]*array2[1]] 
                    )
    
#Conjutate of a quaternion 
def qconj(array):
    return np.array([array[0],-array[1],-array[2],-array[3]])

#transposes a matrix
def trans(M):
    return np.array([M[0].tolist(),M[2].tolist(),M[1].tolist(),M[3].tolist()])
    
#returns  quaternion conjuate of a matrix    
def conjmatrix(M):
    return np.array([qconj(matrix[0]),qconj(matrix[2]),qconj(matrix[1]),qconj(matrix[3])])

#returns quaternionic transpose of a matrix
def qtrans(M):
    val = trans(M)
    return conjmatrix(val)

#matrix multiplication where the entries are quaternions
def matrixmult(M,N):
    v1 = (qmult(M[0],N[0]) + qmult(M[1],N[2])).tolist()
    v2 = (qmult(M[0],N[1]) + qmult(M[1],N[3])).tolist()
    v3 = (qmult(M[2],N[0]) + qmult(M[3],N[2])).tolist()
    v4 = (qmult(M[2],N[1]) + qmult(M[3],N[3])).tolist()
    
    return np.array([v1,v2,v3,v4]) 
    
#matrix multiplied with its quaternionic transpose, we need this to check if a matrix is symplectic

def qt(M):
    return matrixmult(M,qtrans(M))


# In[35]:


T = qmult(U,W)
#qfancy(T)
#qfancy(qconj(H))
#conjmatrix(matrix) 
qtrans(matrix)

qt(matrix)


# In[ ]:





# In[ ]:





# In[ ]:




