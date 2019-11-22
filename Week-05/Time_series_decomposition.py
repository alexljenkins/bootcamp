import numpy as np
from matplotlib import pyplot as plt
#101 numbers between 0 and 10 (inclusive) evenly spaced
x = np.linspace(0,10,101)

# RANDOM NOISE

#uni distribution
y1 = np.random.random(101)
plt.plot(x,y1, label = "uniform noise")

#normal distribution
y2 = np.random.normal(size=101, label = "normal noise")
plt.plot(x,y2)

#random walk
#previous value + the new random number to it
y3 = np.cumsum(y2, label = "random walk")
plt.plot(x,y3)

# TREND (+ noise)
y5 = x*3
plt.plot(x,y5 + y2, label = "trend")


# SEASONALITY
y6 = np.sin(10*x) *10
plt.plot(x,y6, label = "seasonal")

# Meshing it all together
y7 = y5 + y6 + y2 + y3
plt.plot(x,y6, label = "mix")
plt.show()
