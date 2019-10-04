# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:26:10 2019

@author: alexl
"""

ttt = [['.', '.', '.'],
       ['.', 'X', 'O'],
       ['O', 'X', 'O']]

players = ['X', 'O']

line1 = ttt[0]
line2 = ttt[1]
line3 = ttt[2]
line4 = [ttt[0][0],ttt[1][0],ttt[2][0]]
line5 = [ttt[0][0],ttt[1][0],ttt[2][0]]
line6 = [ttt[0][2],ttt[1][2],ttt[2][2]]
line7 = [ttt[0][0],ttt[1][1],ttt[2][2]]
line8 = [ttt[0][2],ttt[1][1],ttt[2][0]]

checks = [line1,line2,line3,line4,line5,line6,line7,line8]

def CheckWinConditions(current_board_state):
    """
    checks if x has 3 in a row
    then check if o has 3 in a row
    """
    
    for xwin in current_board_state:
        if xwin == ['X', 'X', 'X']:# or ['O', 'O', 'O']:
            print("Three in a row! Player X wins!")
                
    for owin in current_board_state:
        if owin == ['O', 'O', 'O']:
            print("Three in a row! Player O wins!")
            

#CheckWinConditions(checks)


#-------------------------------

for (x1,y1),(x2,y2),(x3,y3) in checks:
    if ttt[x1][y1] == ttt[x2][y2] == ttt[x3][y3] and ttt[x1][y1] in 'XO':
        print(f'winner {ttt[x1][y1]}')



























