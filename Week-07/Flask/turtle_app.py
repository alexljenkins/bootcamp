    # -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:15:07 2019

Help from:
https://www.youtube.com/watch?v=-0q_miviUDs

#ajax requests to get and recieve from the backend
"""

import turtle
import math
import random
import numpy as np
import time
import pandas as pd

wn = turtle.Screen()
wn.bgcolor("black")
wn.title("Shopping Sim")
wn.setup(700,700)
# wn.tracer(0) #removes the slow loading of the game
probability_dist = pd.read_csv("C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-08\\prob_dist.csv", sep =",", index_col="location")
dairy = []
spices = []
drinks = []
fruit = []
checkout = []
hallway_upper = []
hallway_lower = []
entrance = []
exit = []

def setup_shop(layout):
    for y in range(len(layout)):
        for x in range(len(layout[y])):
            character = layout[y][x]
            screen_x = -288 + (x*24)
            screen_y = 288 - (y*24)

            #create the walls of the shop
            if character == "X":
                pen.goto(screen_x, screen_y)
                pen.stamp()

                # CURRENTLY COLLISION NOT REQUIRED
                #add coordinates to wall list for collision
                # walls.append((screen_x, screen_y))

            # if character == "C":
            #     customers.append(Customer(screen_x, screen_y))

            if character == "m":
                dairy.append((screen_x, screen_y))
            if character == "s":
                spices.append((screen_x, screen_y))
            if character == "d":
                drinks.append((screen_x, screen_y))
            if character == "f":
                fruit.append((screen_x, screen_y))
            if character == "u":
                hallway_upper.append((screen_x, screen_y))
            if character == "l":
                hallway_lower.append((screen_x, screen_y))
            if character == "o":
                checkout.append((screen_x, screen_y))

            if character == "e":
                entrance.append((screen_x, screen_y))
            if character == "x":
                exit.append((screen_x, screen_y))




#Creates the Mapper tool
class Mapper(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("white")
        self.penup()
        self.speed(0)


print(probability_dist)

class Customer(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.node = "entrance"

        self.shape("square")
        self.color("blue")
        self.penup()
        self.speed(6)
        self.goto((24*9),(24*-16))
        self.next = random.choices(['checkout','dairy','drinks','fruit','spices'],probability_dist.loc[self.node])[0]


    def up(self):
        #moves the customer up
        self.goto(self.xcor(), self.ycor() + 24)

    def down(self):
        #moves the customer up
        self.goto(self.xcor(), self.ycor() - 24)

    def left(self):
        #moves the customer up
        self.goto(self.xcor() - 24, self.ycor())

    def right(self):
        #moves the customer up
        self.goto(self.xcor() + 24, self.ycor())


    def to_hallway(self, goal):
        """
        move customer to a hallway position
        """

        # randomly pick up or down
        self.ymove = np.random.randint(2, size = 1)
        if self.ymove == 0:
            # move up to hallway_upper
            y = random.choice(hallway_upper)[1]
            while self.ycor() < y:
                turtle.ontimer(self.up(), t = 100)
            while self.ycor() > y:
                turtle.ontimer(self.down(), t = 100)

        elif self.ymove == 1:
            # move down to hallway_lower
            random.choice(hallway_lower)
            y = random.choice(hallway_lower)[1]
            while self.ycor() < y:
                turtle.ontimer(self.up(), t = 100)
            while self.ycor() > y:
                turtle.ontimer(self.down(), t = 100)
        self.along_hallway(goal) # start the next set of movements


    def along_hallway(self, goal):
        """
        move customer to a hallway position
        """
        x = goal[0]
        while self.xcor() < x:
            turtle.ontimer(self.right(), t = 100)
        while self.xcor() > x:
            turtle.ontimer(self.left(), t = 100)

        self.to_goal(goal) # start the next set of movements

    def to_goal(self, goal):
        y = goal[1]
        while self.ycor() < y:
            turtle.ontimer(self.up(), t = 100)
        while self.ycor() > y:
            turtle.ontimer(self.down(), t = 100)

    def move(self):

        if self.next == 'dairy':
            # change customer's color
            self.color("blue")
            #move to dairy area "m"
            self.goal = random.choice(dairy)
            self.to_hallway(self.goal)
            #set node to next
            self.node = self.next
            self.next = random.choices(['checkout','dairy','drinks','fruit','spices'],probability_dist.loc[self.node])[0]

        elif self.next == 'spices':
            # change customer's color
            self.color("red")
            #move to spices area "s"
            self.goal = random.choice(spices)
            self.to_hallway(self.goal)
            #set node to next
            self.node = self.next
            self.next = random.choices(['checkout','dairy','drinks','fruit','spices'],probability_dist.loc[self.node])[0]

        elif self.next == 'drinks':
            # change customer's color
            self.color("purple")
            #move to drinks area "d"
            self.goal = random.choice(drinks)
            self.to_hallway(self.goal)
            #set node to next
            self.node = self.next
            self.next = random.choices(['checkout','dairy','drinks','fruit','spices'],probability_dist.loc[self.node])[0]

        elif self.next == 'fruit':
            # change customer's color
            self.color("green")
            #move to fruit area "f"
            self.goal = random.choice(fruit)
            self.to_hallway(self.goal)
            #set node to next
            self.node = self.next
            self.next = random.choices(['checkout','dairy','drinks','fruit','spices'],probability_dist.loc[self.node])[0]

        elif self.next == 'checkout':
            # change customer's color
            self.color("pink")
            #move to checkout area "o"
            self.goal = random.choice(checkout)
            self.to_hallway(self.goal)

            self.goal = random.choice(exit)
            turtle.ontimer(self.down(), t = 100)

            self.along_hallway(self.goal)
            # set node to next
            self.node = self.goal
            # walk off screen
            for i in range(4):
                turtle.ontimer(self.down(), t = 100)
            self.next = None


            # self.next = random.choice(['dairy','spices','drinks','fruit','checkout'])

        else:
            # do nothing on exit
            pass

        # turtle.ontimer(self.move, t = 500)

LAYOUT = [
"XXXXXXXXXXXXXXXXXXXXXXXXX",
"Xuuuuuuuuuuuuuuuuuuuuuu X",
"Xuuuuuuuuuuuuuuuuuuuuuu X",
"X    XX    XX    XX    XX",
"XmmmmXXssssXXddddXX    XX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"XmmmmXXssssXXddddXXffffXX",
"X    XX    XX    XX    XX",
"Xllllllllllllllllllllll X",
"Xllllllllllllllllllllll X",
"X XX XX XX XXXXXXX      X",
"X XX XX XX XXXXXXX      X",
"X XX XX XX XXXXXXX      X",
"XoXXoXXoXXoXXXXXXX      X",
"X                       X",
"XXXXXXXXXXXXXXXXX  xxe  X"]

walls = []
customers = []



if __name__ == '__main__':
    # create map
    pen = Mapper()
    setup_shop(LAYOUT)


    # create customers
    p1 = Customer()
    # p2 = Customer()


    while True:
        p1.move()
        # p2.move()
