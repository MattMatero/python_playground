import matplotlib.pyplot as plt
from collections import Counter

def line_chart(plt):
  years = [1800, 1900, 2000, 2100]
  pop = [1000000000, 1600000000, 6000000000, 9700000000]

  pop_in_billions = [p/1000000000 for p in pop]

  plt.plot(years, pop_in_billions, color='green', marker='o', linestyle='solid')
  plt.title("World population")
  plt.ylabel("Billions of people")
  plt.show()



line_chart(plt)