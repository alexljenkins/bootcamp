# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:12:08 2019
@author: alexjenkins
"""

import pandas as pd
import pylab as plt
import imageio
import os

images = []
fert = pd.read_csv('gapminder_total_fertility.csv', index_col=0)
life = pd.read_excel('gapminder_lifeexpectancy.xlsx', index_col=0)
pop = pd.read_excel('gapminder_population.xlsx', index_col=0)

ncol = [int(x) for x in fert.columns]
fert.set_axis(axis=1, labels=ncol, inplace=True)
sfert, slife, spop = fert.stack(), life.stack(), pop.stack()

d = {'fertility':sfert, 'lifeexp':slife, 'population':spop}
all_gapminder_data = pd.DataFrame(data=d)
stacked_data = all_gapminder_data.stack()
data = stacked_data.unstack(1)

cmap = plt.get_cmap('tab20', lut = len(data.unstack(1))).colors


def graph(year):
    """
    Creates a scatter plot of fertility,
    life expectancy and population for
    each country for a single year
    """
#    year_df = data[year].unstack(1) #removed in place of in-line unstacking below
    
    data[year].unstack(1).plot.scatter('fertility',
                                        'lifeexp',
                                        s=data[year].unstack(1)['population']/1000000,
                                        c=cmap)
    plt.title(f'Life Expectancy for {year}')
    plt.axis([0,10,0,100])

    plt.savefig('temp_image.png')
    plt.close()
    images.append(imageio.imread('temp_image.png'))

for year in range(1960,data.columns[-1]):
    graph(year)
    os.remove('temp_image.png') #kept this outside function, so function is still useful for single use

imageio.mimsave('final.gif', images, fps=20)