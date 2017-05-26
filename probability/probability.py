from __future__ import division
from collections import Counter
import math, random

def random_kid():
  return random.choice(["boy","girl"])

def uniform_pdf(x):
  return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
  "returns the probability that a uniform random variable is less than x"
  if x < 0:   return 0
  elif x < 1: return x 
  else:       return 1

def normal_pdf(x, mu=0, sigma=1):
  sqrt_two_pi = math.sqrt(2*math.pi)
  return (math.exp(-(x-mu)**2/2/sigma**2)/(sqrt_two_pi * sigma))

def plot_normal_pdfs(plt):
  xs = [ x/10.0 for x in range(-50,50) ]
  plt.plot(xs,[normal_pdf(x) for x in xs], '-', label='mu=0,sigma=1') 
  plt.plot(xs,[normal_pdf(x, sigma=2) for x in xs], '-', label='mu=0,sigma=2') 
  plt.plot(xs,[normal_pdf(x, sigma=.5) for x in xs], '-', label='mu=0,sigma=.5') 
  plt.plot(xs,[normal_pdf(x, mu=-1) for x in xs], '-', label='mu=-1,sigma=1') 
  plt.legend()
  plt.show()

def normal_cdf(x,mu=0,sigma=1):
  return (1 + math.erf(x-mu/math.sqrt(2)/sigma)) /2

def plot_normal_cdfs(plt):
  xs = [ x/10.0 for x in range(-50,50) ]
  plt.plot(xs,[normal_cdf(x) for x in xs], '-',label="mu=0,sigma=1")
  plt.plot(xs,[normal_cdf(x, sigma=2) for x in xs], '-',label="mu=0,sigma=2")
  plt.plot(xs,[normal_cdf(x, sigma=.5) for x in xs], '-',label="mu=0,sigma=.5")
  plt.plot(xs,[normal_cdf(x, mu=-1) for x in xs], '-',label="mu=.5,sigma=1")
  plt.legend(loc=4)
  plt.show()

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
  """bin search to find aprox inverse"""
  if mu != 0 or sigma != 1:
    return mu + sigma *inverse_normal_cdf(p,tolerance=tolerance)

  low_z, low_p = -10, 0
  hi_z, hi_p = 10,1

  while hi_z - low_z > tolerance:
    mid_z = (low_z + hi_z)/2
    mid_p = normal_cdf(mid_z)

    if mid_p < p:
      low_z,low_p = mid_z, mid_p
    elif mid_p > p:
      hi_z,hi_p = mid_z, mid_p
    else:
      break

  return mid_z

def bernouli_trial(p):
  return 1 if ranomd.random() < p else 0

def binomial(p,n):
  return sum(bernouli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):
  data = [ binomial(p,n) for _ in range(num_points) ]

  histogram = Counter(data)
  plt.bar([ x - .4 for x in histogram.keys()],
          [ v / num_points for v in histogram.values()],
          .8, color='.75')

  mu = p*n
  sigma = math.sqrt(n*p*(1-p))

  xs = range(mid(data), max(data)+1)
  ys = [ normal_cdf(i + .5, mu,sigma) - normal_cdf(i - .5, mu, sigma) for i in xs]
  plt.plot(xs,ys)
  plt.show()