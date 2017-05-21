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

def histogram(plt):
  grades = [83,50,44,63,83,70,75,90,100,35,15,87,88,95]

  histogram = Counter(grade//10 * 10 for grade in grades)

  plt.bar([x for x in histogram.keys()],
           histogram.values(),
           8)
  plt.axis([-5,105,0,5]) #xaxis bounds/yaxis bounds

  plt.xticks([10*i for i in range(11)])
  plt.xlabel('Grades by letter spread')
  plt.ylabel("Distribution of grades")
  plt.show()


def multi_line_chart(plt):
  variance = [1,2,4,8,16,32,64,128,256]
  bias_squared = variance[:][::-1] #copy and reverse variance
  total_error = [x + y for x,y in zip(variance,bias_squared)]

  y = range(len(variance))

  plt.plot(y, variance, 'g-', label='variance')
  plt.plot(y, bias_squared, 'r-.', label='bias^2')
  plt.plot(y, total_error, 'b:', label='total_error')

  plt.legend(loc=9)
  plt.xlabel("Model complexity")
  plt.title("The bias-variance tradeoff")
  plt.show()


def scatter_plot(plt, eql_axes=False):
  first_exam_grades = [99,90,85,97,80]
  second_exam_grades = [100,85,60,90,70]

  plt.scatter(first_exam_grades, second_exam_grades)
  plt.xlabel("First Test Grades")
  plt.ylabel("Second Test Grades")

  if eql_axes:
    plt.axis("equal")
  
  plt.show()


def pie_chart(plt):
  plt.pie([0.95,0.05], labels=["pacman", "mouth"])
  plt.axis("equal")
  plt.show()
