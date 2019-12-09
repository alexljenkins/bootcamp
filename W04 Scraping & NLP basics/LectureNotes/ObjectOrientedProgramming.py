# -*- coding: utf-8 -*-
"""
OOP is a programming paradign based on the concept of objects

_ at the start of an attribute means "please don't touch"
__ at the start of an attribute means can't be overwritten by subclasses
"""

class Player():
    """
    Creates a player
    """
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.current_bet = 0
        
        
    
    def raise_bet(self, raise_value = 100):
        self.current_bet += raise_value
        self.stack -= raise_value
        
        
    def __repr__(self):
        """
        Good for debugging. Allows to see what the current state of objects are
        """
        
        return f"{self.name} has a stacksize of ${self.stack_size} and current bet is ${self.current_bet}"
    
    pass



Kristian = Player('Kristian', 1000)
David = Player('David', 500)
Alex = Player('Alex', 1000000)

Kristian.raise_bet()
Kristian








