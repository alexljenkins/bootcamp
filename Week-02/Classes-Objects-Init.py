# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:45:52 2019
@author: alexjenkins
Tutorial from:
https://www.youtube.com/watch?v=wfcWRAxRVBA&t=261s
https://www.youtube.com/watch?v=WOwi0h_-dfA
"""

class Robot:
    def __init__(self, name, color, weight):
        """
        This will run the first time a new object is created for the class
        'self' basically refers to the specific object instance
        """
        self.name = name
        self.color = color
        self.weight = weight
        
    def introduce_self(self):
        """
        You must pass 'self' to every function within a class
        """
        print(f"My name is {self.name}")

r1 = Robot('Tom','red',30)
r2 = Robot('Jerry','blue',40)


r1.introduce_self()


#------------------ Creating classes that interact with other classes ------------------#

class Person:
    def __init__(self, n, p, i):
        self.name = n
        self.personality = p
        self.is_sitting = i
        
    def sit_down(self):
        self.is_sitting = True
    
    def stand_up(self):
        self.is_sitting = False

p1 = Person('Alice', 'aggressive', False)
p2 = Person('Becky', 'talkative', True)

#can create additional attributes by just writing them out
p1.robot_owned = r2
p2.robot_owned = r1

#now you can call methods through the attributes
p1.robot_owned.introduce_self() #this grabs the robot that p1 owns, then runs the introduce_self function















