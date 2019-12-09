# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:18:29 2019
@author: alexjenkins
"""

import pandas as pd
from matplotlib import pyplot as plt
import imageio
import os

life = pd.read_excel(r"gapminder_lifeexpectancy.xlsx", index_col=0)
image = []

def life_expectancy_year(year):
    """
    Creates a bar graph of life expectancy for a single year
    """
    
    avglife = round(life[year].mean(),2)
    life.plot.hist(y=year, color='orange', figsize=(15,10), legend=None)
    plt.title(f'Life Expectancy for {year}, Avg. value: {avglife}')
    plt.axis([0,100,0,100])
    plt.savefig(f'{year}.png')
    plt.close()
    image.append(imageio.imread(f'{year}.png'))
    os.remove(f'{year}.png')


for year in life.columns[1:]:
    life_expectancy_year(year)

imageio.mimsave('output.gif', image, fps=20)
