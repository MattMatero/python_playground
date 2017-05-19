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


def bar_chart(plt):
  participants = ["Matt", "John", "Jeff", "Bob"]
  events_won = [3,4,11,7]

  bar_width = [i + .25 for i,_ in enumerate(participants)]

  plt.bar(bar_width, events_won)
  plt.ylabel("# of events won")
  plt.title("Competition Results")

  plt.xticks([i + 0.25 for i,_ in enumerate(participants)], participants)
  plt.show()  


bar_chart(plt)