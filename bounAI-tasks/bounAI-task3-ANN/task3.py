import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("aug_train.csv")

data = data.dropna()

data.loc[data['experience'] == '<1', 'experience'] = 0
data.loc[data['experience'] == '>20', 'experience'] = 21
data.loc[data['last_new_job'] == 'never', 'last_new_job'] = 0
data.loc[data['last_new_job'] == '>4', 'last_new_job'] = 5

data['experience'] = pd.to_numeric(data['experience'])
data['last_new_job'] = pd.to_numeric(data['last_new_job'])


def normalizer(data, cols):
    mean = data[cols].mean()
    data[cols] -= mean
    var = data[cols].var()
    data[cols] /= var
    return data, mean, var


columns = np.array(['training_hours', 'experience', 'last_new_job'])
stats = np.empty((3, 2))
for i in range(0, 3):
    data, stats[i, 0], stats[i, 1] = normalizer(data, columns[i])

final = (data[['city_development_index', 'training_hours', 'experience', 'last_new_job']]).values

categoricals = np.array(['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type'])

for i in categoricals:
    values = np.array(data[i])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    # Makes a column
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    final = np.hstack((final, onehot_encoder.fit_transform(integer_encoded)))

columns = np.array(['city_development_index', 'training_hours', 'experience', 'last_new_job'])

for i in categoricals:
    uniques = np.array(sorted(data[i].unique()))
    columns = np.concatenate((columns, uniques))

