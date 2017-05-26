from __future__ import division
import sys
sys.path.insert(0, '../probability')
from probability import normal_cdf, inverse_normal_cdf
import math,random 

def normal_approximation_to_binomial(n,p):
  """ finds mu and sigma coressponding to binomial(n,p)"""
  mu = p*n
  sigma = math.sqrt(p*(1-p)*n)
  return mu,sigma


#
# Probabilities a normal lies in an interval
#
normal_probability_below = normal_cdf #definition of cdf

def normal_probability_above(lo, mu=0, sigma=1):
  return 1 - normal_cdf(lo,mu,sigma)

def normal_probability_between(lo,hi,mu=0,sigma=1):
  return normal_cdf(hi,mu,sigma) - normal_cdf(lo,mu,sigma)

def normal_probability_outside(lo,hi,mu=0,sigma=1):
  return 1 - normal_probability_between(lo,hi,mu,sigma)

#
# Normal bounds
#

def normal_upper_bound(probability, mu=0, sigma=1):
  """returns the z for which P(Z <= z) = probability"""
  return inverse_normal_cdf(probability,mu,sigma)

def normal_lower_bound(probability,mu=0,sigma=1):
  """returns the z for which P(Z >= z) = probability"""
  return inverse_normal_cdf(1-probability,mu,sigma)

def normal_two_sided_bounds(probability,mu=0,sigma=1):
  """returns the symmetric bounds that contain the specified probability"""
  tail_probability = (1 - probability) / 2
  
  #upper bound should have tail prob above it
  upper_bound = normal_lower_bound(tail_probability, mu, sigma)
  #lower bound should have tail prob below it
  lower_bound = normal_upper_bound(tail_probability, mu,sigma)

  return lower_bound,upper_bound

def two_sided_pvalue(x,mu=0,sigma=1):
  if x >= mu:
    return 2*normal_probability_above(x,mu,sigma)
  else:
    return 2*normal_probability_below(x,mu,sigma)


#
# P hacking
#

def run_experiment():
  return [random.random() < .5 for _ in range(1000)]

def reject_fairness(expermient):
  num_heads = len([flip for flip in experiment if flip])
  return num_heads < 469 or num_heads > 531


#
# Running A/B test
#

def estimated_parameters(N,n):
  p = n / N
  sigma = math.sqrt(p*(1-p)/N)
  return p,sigma

def a_b_test_stats(N_A, n_A, N_B, n_B):
  p_A, sigma_A = estimated_parameters(N_A, n_A)
  p_B, sigma_B = estimated_parameters(N_B, n_B)
  return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)


#
# Bayesian inference
#

def B(alpha,beta):
  """a normalizing constant so that the total probability is 1"""
  return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x,alpha,beta):
  if x < 0 or x > 1:
    return 0
  return x**(alpha - 1) * (1 - x)**(beta - 1) / B(alpha,beta)
