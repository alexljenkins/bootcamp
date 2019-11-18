import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta as timedelta
# sys.path.append('C:\\Users\\alexl\\Documents\\GitPython')
PATH = "C:\\Users\\alexl\\Documents\\GitPython\\cartamon-code\\Week-08\\Data\\"
conn = sqlite3.connect('users.db')
c = conn.cursor()

# drops old table and create new
try:
    c.execute("DROP TABLE data")
    c.execute('''CREATE TABLE data (timestamp TIMESTAMP, customer_no INTEGER, location TEXT, next TEXT);''')
except:
    c.execute('''CREATE TABLE data (timestamp TIMESTAMP, customer_no INTEGER, location TEXT, next TEXT);''')


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
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)

#%%



def full_timeline_creator(df):
    """
    Extends each individual customer in the dataframe to have a minute-by-minute
    account of their location and 'next' location within the store.
    Including a "entrance" field.
    """
    full = pd.DataFrame()

    for customer in df.customer_no.unique():
        # current data for one customer
        current = df[df['customer_no'] == customer]

        # full timeline for that user + 1 initial state
        customer_timeline = pd.date_range(start=current['timestamp'].min() - timedelta(minutes = 1), end=current['timestamp'].max(), freq='T')
        full_df = customer_timeline.to_frame(name='timestamp')

        # merge know values into full timeline for that user
        data = pd.merge(full_df, current, how='left', left_on='timestamp', right_on='timestamp', left_index=True, right_index=False, sort=False)

        # forward fill na data
        data['customer_no'].ffill(inplace=True)
        data['location'].ffill(inplace=True)

        # add an initial state
        # could 'cheat' and just fillna with "entrance" this could speed this up.
        data.loc[data['timestamp'] == data['timestamp'].min(), 'location'] = "entrance"
        data['customer_no'].bfill(inplace=True)

        # create "next" column which represents where the customer moves to next
        data.sort_values(by=['customer_no','timestamp'],inplace=True)
        data['next'] = data['location'].shift(-1)

        data.loc[data['location'] == "checkout", 'next'] = "checkout"
        data['next'].fillna("checkout")

        full = full.append(data, ignore_index=True)

    full['next'].fillna("checkout",inplace=True)

    return full

data = full_timeline_creator(df)

data.to_sql('data', conn, if_exists='append', index=False)
data.head()
#%%
total = data.groupby(['location'])['customer_no'].count()
probability_dist = data.groupby(['location', 'next'])['customer_no'].count()/total

data.isna().sum()
probability_dist['fruit'].sum()




probability_dist = probability_dist.unstack()


def find_probablity_dist(df):

    total = df.groupby(['location'])['customer_no'].count()
    probability_dist = df.groupby(['location', 'next'])['customer_no'].count()/total
    probability_dist = probability_dist.unstack()

    return probability_dist

prob_dist = find_probablity_dist(data)

prob_dist.sum(axis=1)
print(prob_dist)
#%%

#Nb of customers at checkout over time
data.location.unique()
locations = ['entrance','dairy','spices','drinks','fruit','checkout']


a = pd.DataFrame(c.execute('''SELECT location, timestamp, customer_no FROM data WHERE location='checkout' GROUP BY customer_no;''').fetchall())
a
time = a.iloc[:,1].apply(pd.to_datetime)



# df.nth(0).groupby('location').count()
# df.nth(0).groupby(df['location']).count()

# Save (commit) the changes
# conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
# conn.close()
