# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:21:25 2019

@author: alexl

Print a checkers board of #_.
Challenges:
    - 1 print statement
    - no assignments
    - only use one # and _
"""

def CB(x):
    if x % 9 == 0:
        return "\n"
    elif x % 2 == 0:
        return ("#")
    else:
        return ("_")

for i in range (72):
    print (CB(i), end = '')
    

