import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

supermarket= [
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
[0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
[0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
[0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,9,1,0]]


def path_finder(map, xs,ys, xe,ye):

    grid = Grid(matrix=map)
    start = grid.node(xs,ys)
    end = grid.node(xe,ye)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never) #always
    path, runs = finder.find_path(start, end, grid)

    print(f'operations: {runs} with final path length: {len(path)}')
    print(grid.grid_str(path=path, start=start, end=end))

    return path

###############################################################################
#%% Original Supermarket Layout
###############################################################################
path = path_finder(supermarket, 22,24, 4,4)
print(path)
###############################################################################
#%% Random Array with Weights
###############################################################################

random_array = (np.random.rand(26,26) * 10).astype(int)
print(random_array)
path_finder(random_array, 22,24, 4,4)

###############################################################################
#%% Of a Maze Generated Image
###############################################################################
import imageio
image = imageio.imread("Maze_100x100.png")

image_array = (image[:,:,0]/255).astype(int)
path_finder(image_array, 4,4, 20,499)

print(path)
