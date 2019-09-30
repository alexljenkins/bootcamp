# -*- coding: utf-8 -*-
"""
go through the numbers from 1 to 100
if the number is divisible by 3, write fizz:
if the number is divisable by 5, write buzz:
if the number is divisable by 3 and 5, write fizzbuzz:

write the code into a function
add a parameter that allows for other numbers than 100
"""

def FizzBuzz(x):
    for i in range(1,x+1):
        if i % 5 == 0 and i % 3 == 0:
            print('FizzBuzz')
        elif i % 5 == 0:
            print('Buzz')
        elif i % 3 == 0:
            print('Fizz')

#FizzBuzz(int(input("Please enter a integer:")))


def FizzBuzz2(x):
    for i in range(1,x+1):
        p = ""
        if i % 3 == 0:
            p += "Fizz"
        if i % 5 == 0:
            p += "Buzz"
        if p:
            print(p)

FizzBuzz2(int(input("Please enter a integer:")))
