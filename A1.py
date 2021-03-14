#!/usr/bin/env python
# coding: utf-8

# # Assignment A1 [35 marks]
# 
# 

# The assignment consists of 4 exercises. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **discussion** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Numerical Linear Algebra [8 marks]
# 
# **1.1** Using the method of your choice, solve the linear system $Ax = b$ with
# 
# $$ A = \begin{pmatrix}
#           1 &  1 & 0 & 1  \\ 
#          -1 &  0 & 1 & 1  \\ 
#           0 & -1 & 0 & -1  \\ 
#           1 & 0 & 1 & 0 
#         \end{pmatrix}
#         \qquad \text{and} \qquad 
#     b = \begin{pmatrix}
#            5.2 \cr 0.1 \cr 1.9 \cr 0
#         \end{pmatrix},
# $$
# 
# and compute the residual norm $r = \|Ax-b\|_2$. Display the value of $r$ in a clear and easily readable manner.
# 
# **[2 marks]**

# In[17]:


import numpy as np # import the packages that we will be using

A=np.array([[1,1,0,1],[-1,0,1,1],[0,-1,0,-1],[1,0,1,0]]) # define A and b as they are in the question
b=np.array([5.2,0.1,1.9,0])
x = np.linalg.solve(A, b) # a numpy function that will solve the system Ax=b for us 
print("x is equal to", x)

r=((A*x)-b)%2           # calculate the residual norm which is the value of A*x-b in mod 2
print("r is equal to:") 
print(r)


# **1.2** Repeat the same calculations for the matrix
# 
# $$ A = \begin{pmatrix}
#           a &  1 & 0 & 1  \\ 
#          -1 &  0 & 1 & 1  \\ 
#           0 & -1 & 0 & -1  \\ 
#           1 & 0 & 1 & 0 
#         \end{pmatrix}
#         \qquad \text{with} \qquad a \in \{10^{-8}, 10^{-10}, 10^{-12}\}. 
# $$
# 
# Display the value of $r$ for each value of $a$, and avoid repeating (copy+pasting) code.
# 
# **[3 marks]**

# In[20]:


import numpy as np  # import packages


A_1 = np.array([[1e-8, 1, 0, 1],    # define the matrix A1 as A with a=10^-8
              [-1, 0, 1, 1],
              [0, -1, 0, -1],
              [1, 0, 1, 0]],)
b = np.array([5.2, 0.1, 1.9, 0])    # b stays the same as previous part 
x_1=np.dot(np.linalg.inv(A_1), b)   # function from numpy that will solve the equation Ax=b

print('x1 is equal to', x_1)
r_1=((A_1*x_1)-b)%2                 # use this to calculate the residual norm r and display neatly
print("r 1 is equal to:")
print(r_1)

A_2 = np.array([[1e-10, 1, 0, 1],   # repeat for the different values of a
              [-1, 0, 1, 1],
              [0, -1, 0, -1],
              [1, 0, 1, 0]],)
b = np.array([5.2, 0.1, 1.9, 0])  
x_2=np.dot(np.linalg.inv(A_2), b)

print('x2 is equal to', x_2)
r_2=((A_2*x_2)-b)%2
print("r 2 is equal to:")
print(r_2)

A_3 = np.array([[1e-12, 1, 0, 1],
              [-1, 0, 1, 1],
              [0, -1, 0, -1],
              [1, 0, 1, 0]],)
b = np.array([5.2, 0.1, 1.9, 0])
x_3=np.dot(np.linalg.inv(A_3), b)

print('x3 is equal to', x_3)
r_3=((A_3*x_3)-b)%2
print("r 3 is equal to:")
print(r_3)


# **1.3** Summarise and explain your observations in a discussion of no more than $250$ words.
# 
# **[3 marks]**

# 📝 ***Discussion for question 1.3***
# For this question I knew I could find help in the tutorial notebooks. I went back to week * and looked at the notebook for that week. The first step was to solve Ax=b for x which can be done using a function included in the numpy package. The result it gave me was a 1X4 matrix for every value of x which seems correct as they are all consistent in size. The residual norm can be found by taking Ax-b in modulos 2 which for each r is a sqaure, 4X4 matrix. They are again consistent in size as we would expect. Some of my code was taken from the Week 4 Tutorial Notebook. 
# 
# 

# ---
# ## Question 2: Sums [10 marks]
# 
# Consider the sum
# 
# $$
# S_N = \sum_{n=1}^N \frac{2n+1}{n^2(n+1)^2}.
# $$
# 
# **2.1** Write a function `sum_S()` which takes 2 input arguments, a positive integer `N` and a string `direction`, and computes the sum $S_N$ **iteratively** (i.e. using a loop):
# - in the direction of increasing $n$ if `direction` is `'up'`,
# - in the direction of decreasing $n$ if `direction` is `'down'`.
# 
# For instance, to calculate $S_{10}$ in the direction of decreasing $n$, you would call your function using `sum_S(10, 'down')`.
# 
# **[3 marks]**

# In[2]:


def sum_S(N, direction):            # begin by defining a function for our sum, with 2 inputs
    S = 0                            # N and direction. Set S=0 for the begining of the loop
    if direction == "down":         # first condition is when direction is down (decreasing n)   
        for i in range(N, 0, -1): 
            t = (2*i+1)/(i*i*(i+1)**2)  # value of n is put through the formula (t)
            S += t
    elif direction == "up":             # other condition is up (n is increasing)
        for i in range(1, N+1):
            t = (2*i+1)/(i*i*(i+1)**2)   # sub into formula for t again
            S += t
    return S                           # returns the answer

example = sum_S(10, "down")     # test with the value N=10 when n  is decreasing
print(example)                  # the answer is same as what you would manually compute so 
                                 # know we are correct.


# **2.2** The sum $S_N$ has the closed-form expression $S_N = 1-\frac{1}{(N+1)^2}$. We assume that we can compute $S_N$ using this expression without significant loss of precision, i.e. that we can use this expression to obtain a "ground truth" value for $S_N$.
# 
# Using your function `sum_S()`, compute $S_N$ iteratively in both directions, for 10 different values of $N$, linearly spaced, between $10^3$ and $10^6$ (inclusive).
# 
# For each value of $N$, compare the results in each direction with the closed-form expression. Present your results graphically, in a clear and understandable manner.
# 
# **[4 marks]**

# In[14]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt   # we will need these packages for our graph

dN = int((1e6-1e3)/10)        # the linearly spaced values between 10^3 and 10^6

a = "N"                  # labels for N, Ground Truth, up and down
b = "Ground Truth"
c = "up"
d = "down"

print(f"{a: ^7} {b: ^21} {c: ^21} {d: ^21}" )

for N in range(1000, 1000001, dN):   # set a loop for calculating sum_s() in both directions
    S = 1 - (1 / (N+1)**2)          # and for the ground truth for the linearly spaced
    U = sum_S(N, "up")               # values of N
    D = sum_S(N, "down")
    print(f"{N: ^7} {S: .18f} {U: .18f} {D: .18f}")


fig, ax = plt.subplots(1, 1, figsize=(7,9))
    
ax.plot(dN, S, label = "Ground Truth")
ax.plot(dN, U, label = "Up")
ax.plot(dN, D, label = "Down")
ax.set_title('Results')

ax.legend()     
plt.show()



# **2.3** Describe and explain your findings in no more that $250$ words. Which direction of summation provides the more accurate result? Why?
# 
# **[3 marks]**

# 📝 ***Discussion for question 2.3***
# 
# After analysing my results and the graph, I can see that the direction down is more accurate than the direction up. This is clear as the results for each direction down are very close or even sometimes equal to the ground truth value. As for the up direction, we can see from the graph that the result from the up direction converges to 0.999999999992634891 as the value of N increases which makes it less accurate for larger values of N. Code for the graph taken from Week 5 lectures.
# 
# 

# ---
# ## Question 3: Numerical Integration [10 marks]
# 
# For integer $k \neq 0$ consider the integral 
# 
# $$
# I(k) = \int_0^{2\pi}  x^4 \cos{k x} \ dx = \frac{32\pi^3}{k^2} - \frac{48\pi}{k^4} \ .
# $$
# 
# **3.1** Write a function `simpson_I()` which takes 2 input arguments, `k` and `N`, and implements Simpson's rule to compute and return an approximation of $I(k)$, partitioning the interval $[0, 2\pi]$ into $N$ sub-intervals of equal width.
# 
# **[2 marks]**

# In[ ]:


import numpy as np    # import packages needed
import math

def simpson_I(f, a, b, N):     # we begin by defining the function for the Simpson Rule
    h = ((b-a)/(N-1))         # which has inputs f (the function), a (lower limit), b
    I = 0                     # (the upper limit) and N (the number of nodes)
    x = a+h
    for i in range(0, N/2 + 1):   # loop for when i is N/2 +1
        I += 4*f(x)
        x += 2*h
        
    x = a + 2*h
    for i in range(1,N/2):       # loop for when i is N/2
        I += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+k)   
 

def f(x):
    return (x**4*(np.cos(k*x)))  # define the function we want to integrate
    

I_approx = simpson_I(f, 0, (2*np.pi), 1000)   # use our function for the simpson rule on
                                              # the function f(x)
print(f'{I_approx:.6f}')   # print the result


# **3.2** For $k = 1$, and for $\varepsilon \in \{10^{-n} \ |\  n \in \mathbb{N}, 3 \leqslant n \leqslant 8\}$, determine the number $N_{\text{min}}$ of partitions needed to get the value of the integral $I(1)$ correctly to within $\varepsilon$. 
# 
# **[2 marks]**

# In[ ]:





# **3.3** Repeat your calculations from **3.2** for $k \in \{2^{n}\ |\ n \in \mathbb{N}, n \leqslant 6\}$. 
# 
# **[2 marks]**

# In[ ]:





# **3.3** Present your results graphically by plotting 
# 
# (a) the number of terms $N_{\text{min}}$ against $\varepsilon$ for fixed $k$, 
# 
# (b) the number of terms $N_{\text{min}}$ against $k$ for fixed $\varepsilon$.
# 
# You should format the plots so that the data presentation is clear and easy to understand.
# 
# **[2 marks]**

# In[ ]:





# **3.4** Discuss, with reference to your plot from 3.3, your results. Your answer should be no more than $250$ words.
# 
# **[2 marks]**

# 📝 ***Discussion for question 3.4***
# 
# There is an error in my code for 3.1 that prevents it from running. The code should take the function I defined as f(x) and perform the simpson rule on it to integrate it. The simpson rule can be defined as Δx/3(y_0+4y_1+2y_2+4y_3+2y_4+...+4y_n-1+y_n) where Δx=b-a/n, n must be even. Code from week 6 lectures and notebook.
# 
# 
# 

# ---
# ## Question 4: Numerical Derivatives [7 marks]
# 
# Derivatives can be approximated by finite differences in several ways, for instance
# 
# \begin{align*}
#         \frac{df}{dx} & \approx \frac{f(x+h) - f(x)}{h} \\
#         \frac{df}{dx} & \approx \frac{f(x) - f(x-h)}{h}  \\
#         \frac{df}{dx} & \approx \frac{f(x+h) - f(x-h)}{2h} \ . 
# \end{align*}
# 
# Assuming $f$ to be differentiable, in the limit $h \to 0$, all three expressions are equivalent and exact, but for finite $h$ there are differences. Further discrepancies also arise when using finite precision arithmetic.
# 
# **4.1**
# Estimate numerically the derivative of $f(x) = \cos(x)$ at $x = 1$ using the three expressions given above and different step sizes $h$. Use at least 50 logarithmically spaced values $h \in [10^{-16}, 10^{-1}]$.
# 
# **[2 marks]**

# In[5]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt # we will need this package for our graph

def f(x):
    return np.cos(x)     # define the function f(x) which we have been asked to differentiate

def f_prime(x):
    return -1*np.sin(x)   # define f_prime as -sin(x) as this is the exact integral for cos(x)
                          # we will use this to compare to our estimated values for the derivative

def derivative_1(f, h, x):   # this is the forward difference formula for differentiation
    D1 = (f(x+h)-f(x))/h      # it inputs the function f, evaluated at point x, with h representing 'delta x'
    return D1                 # the output of the function is an estimate for the derivative

def derivative_2(f, h, x):   # this defenition is for the backward differences formula
    D2 = (f(x)-f(x-h))/h      # same input values as previous defenition
    return D2                 # output value is another estimate for the derivative, it will be different to the previous

def derivative_3(f, h, x):     # the final formula needed is the central differences formula for derivatives
    D3 = (f(x+h)-f(x-h))/2*h
    return D3

x0 = 1                              # as stated in question, we are evaluating the derivative at the point x=1
h = np.logspace(1e-16, 1e-1, 50)     # we use this numpy function to evaluate the derivative at 50 logarithmically spaced values of h
d_approx1 = derivative_1(f, h, x0)   
d_approx2 = derivative_2(f, h, x0)   # evaluate for the three functions
d_approx3 = derivative_3(f, h, x0)
d_exact = f_prime(x0)      # evaluate for the exact value of the derivative
print('the exact value of the derivative is', d_exact)
#print(d_approx1)
#print(d_approx2)      # for presentation purposes these have been changed to comments
#print(d_approx3)      # removing the "#" will print the 150 values of approximated derivatives

abs_diff1 = np.abs(d_exact - d_approx1)
#print(abs_diff1)                       # the absolute differences can be displayed by removing the '#'                        

abs_diff2 = np.abs(d_exact - d_approx2)  # the values are plotted in the next part of the question
#print(abs_diff2)

abs_diff3 = np.abs(d_exact - d_approx3)
#print(abs_diff3) 
    


# **4.2**
# Display the absolute difference between the numerical results and the
# exact value of the derivative, against $h$ in a doubly logarithmic plot. 
# You should format the plot so that the data presentation is clear and easy to understand.
# 
# **[2 marks]**

# In[17]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt   # import necessary packages

x = np.logspace(1e-16, 1e-1, 50)             # compute the graph
fig, ax = plt.subplots(1, 1, figsize=(8,8))   # choose appropriate size

# we will need three different lines on our graph, each representing the absolute difference
# values abs_diff1, abs_diff2 and abs_diff3. We assign the appropriate labels to each.
ax.plot(x, abs_diff1, label = 'Forward Difference error') 
ax.plot(x, abs_diff2, label = 'Backward Difference error')
ax.plot(x, abs_diff3, label = 'Central Difference error')   
ax.set_title('Errors')

ax.legend() # add a legend to make the graph readable
plt.show()  # display graph


# **4.3**
# Describe and interpret your results in no more than 250 words.
# 
# *Hint: run the code below.*
# 
# **[3 marks]**

# In[8]:


h = 1e-14
print(1 + h - 1)
print((1 + h - 1)/h)


# 📝 ***Discussion for question 4.3***
# My graph shows that each method for the approximation of the derivative has a different value of error. The forward differences method has a consistent value for the errors which can be seen with the almost straight line on the graph. This was already obvious to me as when I printed the 50 values for d_approx1 they were consistent and all were between about -0.93 to -0.95 no matter the value of h.
# 
# The backward differences method gave a clear relationship with its error as can be seen on the graph. When looking at its values for the absolute difference abs_diff2 we can see that it gradually increases as h increases hence meaning the approximation is getting less and less accurate. 
# 
# The central differences error line on the graph is interesting as we can see it decreasing towards zero then begin to increase. This tells us that the error gets smaller and smaller until we get the best value of h needed for the approximation. This h value gives us a really small error (almost 0). Then as the h value increases the estimation starts become less accurate and the error starts to increase. 
# 
# In summary, my results are telling me that the forward differences value seems consistently most accuarate as the errors are consistent for each value of h. However we get our most accurate estimation from the central differences method but only for one value of h.
# Code for graph from week 5 lectures. 
# 

# In[ ]:




