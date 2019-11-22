"""
Could do States and number of teachers/schools trained
as a proportion of the market size of that state.

To see what state you're doing well in based on market size.
Where is the biggest potencial?

"""

import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
import pylab
from itertools import product
import numpy as np
rand = np.random.random
speaks_mul_foreign_languages = list(product(['male', 'female'], ['yes', 'no']))
index = pd.MultiIndex.from_tuples(speaks_mul_foreign_languages, names=['male', 'female'])
data = pd.Series(rand(4), index=index)
mosaic(data, gap=0.01, title='Who knows multiple foregin languages? - Mosaic Chart')
pylab.show()
