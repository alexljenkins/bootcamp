# -*- coding: utf-8 -*-
"""
Scissors, Paper, Rock
@author: alexl
"""

def game():
    user_pick = input('Pick scissors, paper or rock:')
    if user_pick == 'scissors':
        print ('ROCK!')
        win()
    elif (user_pick == 'paper'):
        print ('SCISSORS!')
        win()
    elif (user_pick == 'rock'):
        print ('PAPER!')
        win()
    else:
        print (f'{user_pick} is cheating \nYou must be playing a different game...')
        game()

def win():
    play_again = input('I win.... Want to play again? y/n:')
    if play_again == "yes" or play_again == "y":
        game()
    else:
        print ('You\'re no fun! bye bye')

print ('Let\'s play a game....')        
game()








#------------------------Attempt 2------------------------#
game_states = ['paper','scissors','rock','paper']
move = ''

def game2(move):
    if move in game_states:
        return game_states[game_states.index(move)+1]
    else:
        return (f'{move} is cheating \nYou must be playing a different game...')    

#while move != 'stop':
#    move = input('Pick scissors, paper or rock?:')
#    winning_move = game2(move)
#    print(winning_move)