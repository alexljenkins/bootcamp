# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:53:30 2019
@author: alexl
"""
import random

board = [['.', '.', '.'],
         ['.', '.', '.'],
         ['.', '.', '.']]

players = ['X', 'O']
player = players[round(random.uniform(0, 1))]

print(player)
legal_moves = ['00','01','02','10','11','12','20','21','22']


def PlaceMove(player, move):
    """
    Checks move is valid, and places it on the field.
    Changes player over and runs that player's turn.
    """
    
    if move in legal_moves:
        board[int(move[0])][int(move[1])] = player
        legal_moves.remove(move)
        
        CheckWinConditions(player)
        CheckDraw()
        player = SwapPlayer(player)
        PlayerTurn(player)
    
    elif move == 'exit':
        raise SystemExit
    else:
        print('That was an illegal move. Try again!')
        player = PlayerTurn(player)


def PrintBoardState():
    """
    Prints the current board state
    """
    print(board[0])
    print(board[1])
    print(board[2])

    
def PlayerTurn(player):
    """
    Asks current player what their move is.
    """
    PrintBoardState()
    move = input(f'Player {player}, please select a COLUMN then ROW from 0-2:')
    
    PlaceMove(player, move)
    
    
def SwapPlayer(player):
    """
    Swap players around
    """
    if player == 'X':
        return 'O'
    else:
        return 'X'


def CheckWinConditions(player):
    """
    checks if player has 3 in a row
    """
    #Update row checks
    check1 = board[0]
    check2 = board[1]
    check3 = board[2]
    #Update column checks
    check4 = [board[0][0],board[1][0],board[2][0]]
    check5 = [board[0][0],board[1][0],board[2][0]]
    check6 = [board[0][2],board[1][2],board[2][2]]
    #Update diagonal checks
    check7 = [board[0][0],board[1][1],board[2][2]]
    check8 = [board[0][2],board[1][1],board[2][0]]
    
    checks = [check1,check2,check3,check4,check5,check6,check7,check8]

    for winner in checks:
        if winner == [f'{player}', f'{player}', f'{player}']:
            print(f"Three in a row! Player {player} wins!\nFinal board state:")
            PrintBoardState()
            raise SystemExit


def CheckDraw():
    """
    Checks if the board is full and announces a draw
    """
    if len(legal_moves) == 0:
        print('Game is a draw!\nFinal board state:')
        PrintBoardState()
        raise SystemExit


PlayerTurn(player)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        