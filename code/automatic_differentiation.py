import torch

# For example: y = 2x^T x 
x = torch.arange(4.0)
x

# Before we even calculate the gradient of y with respect to x,
# we will need a place to store it. It is important that we do
# not allocate new memory every time we take a derivative with
# respect to a parameter because we will often update the same
# parameters thousands or millions of times and could quickly
# run out of memory. Note that a gradient of a scalar-valued
# function with respect to a vector x is itself vector-valued
# and has the same shape as x.
x.requires_grad_(True)
x.grad                          # default value is None

y = 2 * torch.dot(x, x)
y                               # scalar

# calculate the gradient of y 
y.backward()

x.grad

# y' = 4x
x.grad == 4 * x



# PyTorch accumulates the gradient in default, we need to clear the previous values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad



# Technically, when y is not a scalar, the most natural
# interpretation of the differentiation of a vector y
# with respect to a vector x is a matrix. For higher-order
# and higher-dimensional y and x, the differentiation
# result could be a high-order tensor.

# More often when we are calling backward on a vector,
# we are trying to calculate the derivatives of the
# loss functions for each constituent of a batch of
# training examples. Here, our intent is not to
# calculate the differentiation matrix but rather the
# sum of the partial derivatives computed individually
# for each example in the batch.
x.grad.zero_()
y = x * x
y                               # tensor
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate

# y.backward(torch.ones(len(x))) equivalent to the following
y.sum().backward()
x.grad


# Sometimes, we wish to move some calculations outside of the recorded
# computational graph. For example, say that y was calculated as a
# function of x, and that subsequently z was calculated as a function
# of both y and x. Now, imagine that we wanted to calculate the gradient
# of z with respect to x, but wanted for some reason to treat y as a
# constant, and only take into account the role that x played after y
# was calculated.

"""
Here, we can detach y to return a new variable u that has the same value
as y but discards any information about how y was computed in the
computational graph. In other words, the gradient will not flow backwards
through u to x. Thus, the following backpropagation function computes the
partial derivative of z = u * x with respect to x while treating u as a
constant, instead of the partial derivative of z = x * x * x with respect
to x.
"""
x.grad.zero_()
y = x * x
y
u = y.detach()
u
z = u * x
z
z.sum().backward()
x.grad == u
