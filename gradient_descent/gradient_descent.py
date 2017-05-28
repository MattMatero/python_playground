from __future__ import division
from collections import Counter
import sys
sys.path.insert(0,'../linear_algebra')
from linear import distance, vector_sub, scalar_mult
import math, random

def sum_of_squares(v):
  return sum([v_i**2 for v_i in v])

def difference_quotient(f,x,h):
  return (f(x+h) - f(x)) / h

def plot_estimated_deriviative():

  def square(x):
    return x*x

  def derivative(x):
    return 2*x

  derivative_estimate = lambda x: difference_quotient(square,x,h=0.00001)

  import mathplotlib.pyplot as plot
  x = range(-10,10)
  plt.plot(x,map(derivative,x), 'rx')
  plt.plot(x,map(derivative_estimate,x), 'b+')
  plot.show()

def partial_difference_quotient(f,v,i,h):
  w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

  return (f(w) - f(v)) / h

def estimate_gradient(f,v,h=0.00001):
  return [partial_difference_quotient(f,v,i,h) for i,_ in enumerate(v)]

def step(v,direction,step_size):
  return [v_i + step_size * direction_i for v_i, direction_i in zip(v,direction)]

def safe(f):

  def safe_f(*args, **kwargs):
    try:
      return f(*args,**kwargs)
    except:
      return float('inf') #infinity

  return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
  step_sizes = [100,10,1,.1,.01,.001,.0001,.00001]

  theta = theta_0
  target_fn = safe(target_fn)
  value = target_fn(theta)

  while True:
    gradient = gradient_fn(theta)
    next_thetas = [step(theta,gradient,-step_size) for step_size in step_sizes]

    next_theta = min(next_thetas, key=target_fn)
    next_value = target_fn(next_theta)

    if abs(value - next_value) < tolerance:
      return theta
    else:
      theta,value = next_theta,next_value

def negate(f):
  return lambda *args, **kwargs: -f(*args,**kwargs)

def negate_all(f):
  return lambda *args, **kwargs: [-y for y in f(*args,**kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
  return minimize_batch(negate(target_fn), negate_all(gradient_fn), theta_0, tolerance)

def in_random_order(data):
  indices = [i for i,_ in enumerate(data)]
  random.shuffle(indices)
  for i in indices:
    yield data[i]

def minimize_stochastic(target_fn,gradient_fn,x,y,theta_0,alpha_0=.01):
  data = zip(x,y)
  theta = theta_0
  alpha = alpha_0
  min_theta, min_value = None, float("inf") #min so far
  iterations_with_no_improvement = 0

  while iterations_with_no_improvement < 100:
    value = sum(target_fn(x_i,y_i,theta) for x_i, y_i in data)

    if value < min_value:
      min_theta, min_value = theta, value
      iterations_with_no_improvement = 0
      alpha = alpha_0
    else:
      iterations_with_no_improvement += 1
      alpha *= .9

    for x_i, y_i in in_random_order(data):
      gradient_i = gradient_fn(x_i, y_i, theta)
      theta = vector_sub(theta,scalar_mult(alpha,gradient_i))

  return min_theta

def maximize_stochastisc(target_fn,gradient_fn,x,y,theta_0,alpha_0=.01):
  return minimize_stochastic(negate(target_fn), negate_all(gradient_fn), x,y,theta_0,alpha_0)

