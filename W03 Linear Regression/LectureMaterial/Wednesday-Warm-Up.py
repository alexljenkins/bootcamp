# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:16:43 2019

@author: alexl
"""

import turtle

#
#
#for i in range(0,500,5):
#    turtle.forward(i)
#    turtle.left()
#
#
#
#turtle.done()
#turtle.bye()
##x = 1
##def move(x, i):
##    x+=x+i
##    turtle.forward(x)
##    return x
##
##
##
a = 1
b = 2

def tu(a, b):
    temp = b
    b = b + a
    a = temp
    turtle.forward(b**.02)
    turtle.left(90/b**0.1)
    return a, b

for i in range(500):
    a, b = tu(a,b)
    
turtle.done()
turtle.bye()


fibo_nr = [1,1,2,3, 5, 8, 13, 21, 34,55]  #Fibonacci numbers this could be calculated instead

def draw_square(side_length):  #Function for drawing a square
    for i in range(4):
        forward(side_length)
        right(90)

nr_squares=len(fibo_nr)

factor = 3                        #Enlargement factor
penup()
goto(50,50)                  #Move starting point right and up
pendown()
for i in range(nr_squares):
    draw_square(factor*fibo_nr[i]) #Draw square
    penup()                        #Move to new corner as starting point
    forward(factor*fibo_nr[i])
    right(90)
    forward(factor*fibo_nr[i])
    pendown()
        
penup()
goto(50,50)       #Move to starting point
setheading(0)   #Face the turtle to the right
pencolor('red')
pensize(3)
pendown()
#Draw quartercircles with fibonacci numbers as radius
for i in range(nr_squares):
    circle(-factor*fibo_nr[i],90)  # minus sign to draw clockwise
