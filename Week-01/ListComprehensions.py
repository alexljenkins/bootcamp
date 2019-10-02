# -*- coding: utf-8 -*-
"""
Let's learn about list comprehensions!
You are given three integers, x,y,z, representing the dimensions of a cuboid
along with an integer.
You have to print a list of all possible coordinates given by i,j,k
on a 3D grid where the sum of
x,y,z is not equal to n.
Print the list in lexicographic increasing order.
Question found: https://www.hackerrank.com/challenges/list-comprehensions/problem?isFullScreen=true
@author: alexjenkins
"""

x = int(input("Value for x:"))
y = int(input("Value for y:"))
z = int(input("Value for z:"))
n = int(input("Value for n:"))

cuboids = []

for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k != n:
                cuboids.append([i,j,k])
            else:
                continue
print(cuboids)