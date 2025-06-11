#!/usr/bin/env python
# get_ipython().system('pip freeze | grep scikit-learn')
# get_ipython().system('python -V')

import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# Set up homework submission
year = 2023
month = 5

df = read_data('yellow_tripdata_2023-05.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)




# mean predicted duration
duration_mean = y_pred.mean()
print("Predicted mean: ",duration_mean)
# standard deviation of predicted duration
duration_std = y_pred.std()
print(duration_std)

# # Question 2
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# df


# # In[24]:


# df_result = pd.DataFrame()
# df_result['ride_id'] = df['ride_id']
# df_result['predicted_duration'] = y_pred


# # In[17]:


# output_file = f'yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'


# # In[25]:


# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )


# # In[26]:


# get_ipython().system('ls -lh yellow_tripdata_2023-03_predictions.parquet')


# # In[21]:


# get_ipython().system('ls')


# # In[1]:


# get_ipython().system('jupyter nbconvert --to=starter.ipynb')


# # In[ ]:




