import torch

# range
x = torch.arange(12)
x

# get the shape of the tensor
x.shape

# total number of elements in a tensor
x.numel()

# reshape the shape of tensor
X = x.reshape(3, 4)
X

# make zero tensor
torch.zeros((2,3,4))

# make unit tensor
torch.ones((2,3,4))

# specified tensor with random elements
torch.randn(3,4)

# convert list to tensor
torch.tensor([1,2,3])


# operation
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
x+y                            
x-y
x*y
x/y
x**y

# exponentialion
torch.exp(x)


# tensor concaternate
X = torch.arange(12).reshape((3,4))
X
Y = torch.arange(12).reshape((3,4))
Y
torch.cat((X,Y), dim=0)
torch.cat((X,Y), dim=1)

# sum all elements
X.sum()


# broadcasting mechanism
# 1. expand one or both arrays by copying elements appropriately so that after this transforma- tion, the two tensors have the same shape
# 2. carry out the elementwise operations on the resulting arrays
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
a
b
a+b


# index and slicing
X
X[-1]
X[1:3]
X[1,2] = 9
X
X[0:2, :] = 100
X

# Saving memory
# Running operations can cause new memory to be allocated to host results.
before = id(Y)
Y = Y + X
id(Y) == before
# False


before = id(Y)
Y += X
id(Y) == before
# True


# convert to numpy
A = X.numpy()
B = torch.tensor(A)
type(A)
type(B)


# size-1 tensor to scalar
a = torch.tensor([1])
a.item()

# reduction sum
X
X.sum()

# reduction mean
X = torch.tensor(X, dtype=float)
X.mean()

# num of elements
X.numel()


# num-reduction num
sum_A = X.sum(axis=1, keepdim=True)
sum_A


# dot product
x
y
torch.dot(x,y)

# matrix-vector product
A = torch.arange(12, dtype=float).reshape((3,4))
x = torch.arange(4, dtype=float)
torch.mv(A,x)

# matrix multiplication
A = torch.ones(4,4)
B = torch.ones(4,3)
torch.mm(A,B)


# norm
a = torch.tensor([3.0, -4.0])
torch.norm(a)

# abs
torch.abs(a)


