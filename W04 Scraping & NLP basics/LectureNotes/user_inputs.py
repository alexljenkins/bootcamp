import numpy as np

"""
First Method of getting user input
"""
def calculate_standard_deviation(x, y):
    x = x[:y]
    result = np.std(x)
    print(result)


# array = list(input("please give me an array:\n"))
# limit = int(input('tell me where to chop'))
#
# calculate_standard_deviation(array, limit)


"""
Second Method of getting user input
"""

#takes inputs during initial runtime
#returns them as a list
import sys
arg1 = sys.argv

print(arg1)


"""
Method of only running code if it's excicuted in termal from the original file
"""

if __name__ == '__main__':


from tqdm import tqdm
for i in tqdm(range(10000000000), desc= "description text"):
    pass

from pyfiglet import Figlet
f = Figlet(font='slant')
print(f.renderText('Hellow World'))
