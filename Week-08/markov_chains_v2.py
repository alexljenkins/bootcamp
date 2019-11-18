import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns

# sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
PATH = "C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-08\\Data\\"
conn = sqlite3.connect('users.db')
c = conn.cursor()

# drops old table and create new
try:
    c.execute("DROP TABLE data")
    c.execute('''CREATE TABLE data (timestamp TIMESTAMP, customer_no INTEGER, location TEXT);''')
except:
    c.execute('''CREATE TABLE data (timestamp TIMESTAMP, customer_no INTEGER, location TEXT);''')


#%% fill table with data
MON = pd.read_csv(PATH + 'monday.csv',
                   sep = ';', header = 0, names = ['timestamp', 'customer_no', 'location'])
MON['customer_no'] = MON['customer_no'] + 10000
TUE = pd.read_csv(PATH + 'tuesday.csv',
                   sep = ';', header = 0, names = ['timestamp', 'customer_no', 'location'])
TUE['customer_no'] = TUE['customer_no'] + 20000
WED = pd.read_csv(PATH + 'wednesday.csv',
                   sep = ';', header = 0, names = ['timestamp', 'customer_no', 'location'])
WED['customer_no'] = WED['customer_no'] + 30000
THU = pd.read_csv(PATH + 'thursday.csv',
                   sep = ';', header = 0, names = ['timestamp', 'customer_no', 'location'])
THU['customer_no'] = THU['customer_no'] + 40000
FRI = pd.read_csv(PATH + 'friday.csv',
                   sep = ';', header = 0, names = ['timestamp', 'customer_no', 'location'])
FRI['customer_no'] = FRI['customer_no'] + 50000

df = pd.DataFrame()
df = df.append([MON,TUE,WED,THU,FRI], ignore_index=True)
df.to_sql('data', conn, if_exists='append', index=False)

#%%

# print(pd.DataFrame(c.execute('''SELECT * FROM data;''').fetchall(),
#             columns = ['timestamp', 'customer_no', 'location']).head())



#%%

# checkout = pd.DataFrame()
# checkout['customer_no'] = pd.Series(df['customer_no'].unique())
# checkout['timestamp'], checkout['location'] = 0, "checkout"
# df4 = pd.concat([df, checkout],sort=False).sort_values('customer_no')
# df3 = df3.sort_values(by='customer_no')
#
# df3 = df3.sort_values(by='customer_no')

df.sort_values(by=['customer_no','timestamp'],inplace=True)
df['next'] = df['location'].shift(-1)
df.head()

df.loc[df['location'] == "checkout", 'next'] = "checkout"

#%%

df2 = df.groupby(df['customer_no'])

sns.distplot(df2['location'].count(),bins=np.arange(0,18), hist_kws=dict(ec="k"))


df2.nth(0).groupby('location').count()
df2.nth(0).groupby(df['location']).count()

# shift function for pandas df



df.pivot(index = df.groupby(df['customer_no']), columns='location', values='timestamp')

#%%

def from_finder(df):
    individuals = df.groupby(customer_no)










# Save (commit) the changes
# conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
# conn.close()
