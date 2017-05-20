from __future__ import division
import re, math, random
from collections import defaultdict, Counter
from functools import partial


#Some basic operations on vectors

def vector_add(v,w):
  return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_sub(v,w):
  return[v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
  return reduce(vector_add, vectors)

def scalar_mult(c, v):
  return [c*v_i for v_i in v]

#calculates mean of each 'row' from list of vectors
def vector_mean(vectors):
  n = len(vectors)
  return scalar_mult(1/n, vector_sum(vectors))

def dot(v,w):
  return sum(v_i * w_i for v_i, w_i in zip(v,w))

def sum_of_squares(v):
  return dot(v,v)

def magnitude(v):
  return math.sqrt(sum_of_squares(v))

def squared_distance(v,w):
  return sum_of_squares(vector_sub(v,w))

def distance(v,w):
  return math.sqrt(squared_distance(v,w))



#Matrix operations

def shape(A):
  num_rows = len(A)
  num_cols = len(A[0]) if A else 0
  return num_rows, num_cols

def get_row(A,i):
  return A[i]

def get_col(A,j):
  return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
  return [ [entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows) ]

def is_diagonal(i,j):
  return 1 if i == j else 0

#makes an identity matrix
ident_mat = make_matrix(3,3, is_diagonal)

def matrix_add(A,B):
  if shape(A) != shape(B):
    raise ArithmeticError("cannot add matrices of different shapes")

  num_rows, num_cols = shape(A)
  def entry_fn(i,j): return A[i][j] + B[i][j]

  return make_matrix(num_rows, num_cols, entry_fn)

def matrix_mult(A,B):
  A_rows,A_cols = shape(A)
  B_rows,B_cols = shape(B)
  if A_cols != B_rows:
    raise ArithmeticError("cannot multiply matrices with unequal rows and cols")


  def entry_fn(i,j): return dot(get_row(A,i), get_col(B,j))

  return make_matrix(A_rows, B_rows, entry_fn)


#some practice/examples
def fill_matrix(i,j):
  return i*j + 1 

matrixA = make_matrix(3,3,fill_matrix)
matrixB = make_matrix(3,3, fill_matrix)

print matrixA
print matrixB

matrixC = matrix_add(matrixA, matrixB)

print matrixC

matrixD = matrix_mult(matrixA, matrixB)

print matrixD
