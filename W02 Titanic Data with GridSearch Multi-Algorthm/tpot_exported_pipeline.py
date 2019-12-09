import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('Data/export_all.csv', sep=',', dtype=np.float64)
tpot_predict = pd.read_csv('Data/exported_test.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Survived', axis=1).values

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Survived'].values, random_state=42)

# Average CV score on the training set was:0.8835122623855017
exported_pipeline = RandomForestClassifier(bootstrap=True,
                                           criterion="gini",
                                           max_features=0.1,
                                           min_samples_leaf=5,
                                           min_samples_split=14,
                                           n_estimators=100)
exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)




#for complete dataset
exported_pipeline.fit(features, tpot_data['Survived'])
results2 = exported_pipeline.predict(tpot_predict)
print(results2)
#export = pd.DataFrame(results2, columns = ['Survived']) ##################
#export.to_csv('Data/tpot_attempt2.csv')