#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Load the datasets
trips_by_distance = pd.read_csv("Trips_by_Distance.csv")
trips_full_data = pd.read_csv("Trips_Full Data.csv")


# In[3]:


# Display the first few rows of each dataset to understand their structure
print(trips_by_distance.head())
print(trips_full_data.head())


# In[4]:


# Convert 'Date' column to datetime format
trips_by_distance['Date'] = pd.to_datetime(trips_by_distance['Date'], format='%m/%d/%Y')
trips_full_data['Date'] = pd.to_datetime(trips_full_data['Date'])

# Verify the conversion
print(trips_by_distance[['Date']].head())
print(trips_full_data[['Date']].head())


# In[5]:


# Summing 'Population Staying at Home' and 'Population Not Staying at Home' for each date
home_vs_travel_summary = trips_by_distance.groupby('Date').agg({
    'Population Staying at Home': 'sum',
    'Population Not Staying at Home': 'sum'
}).reset_index()

print(home_vs_travel_summary.head())


# In[6]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(home_vs_travel_summary['Date'], home_vs_travel_summary['Population Staying at Home'], label='Staying at Home')
plt.plot(home_vs_travel_summary['Date'], home_vs_travel_summary['Population Not Staying at Home'], label='Not Staying at Home', alpha=0.7)
plt.title('Population Staying at Home vs. Not Staying at Home Over Time')
plt.xlabel('Date')
plt.ylabel('Population')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[7]:


# Calculate the daily change in population staying at home and not staying at home
home_vs_travel_summary['Change in Staying at Home'] = home_vs_travel_summary['Population Staying at Home'].diff()
home_vs_travel_summary['Change in Not Staying at Home'] = home_vs_travel_summary['Population Not Staying at Home'].diff()

# Look for large changes which might be outliers or significant events
large_changes_home = home_vs_travel_summary[abs(home_vs_travel_summary['Change in Staying at Home']) > home_vs_travel_summary['Change in Staying at Home'].quantile(0.95)]
large_changes_not_home = home_vs_travel_summary[abs(home_vs_travel_summary['Change in Not Staying at Home']) > home_vs_travel_summary['Change in Not Staying at Home'].quantile(0.95)]

print("Significant changes in population staying at home:")
print(large_changes_home)

print("\nSignificant changes in population not staying at home:")
print(large_changes_not_home)


# In[8]:


pip install joblib


# In[9]:


from joblib import Parallel, delayed

# Assuming that each 'Number of Trips' column is already directly available in the DataFrame
# and does not need to be calculated from a range, we can simplify the function.

def calculate_average_trips(data, column_name):
    # Check if the column exists
    if column_name in data.columns:
        return data[column_name].mean()
    else:
        return None  # If the column doesn't exist, return None

# Create a list of column names based on the expected pattern in your DataFrame
column_names = ['Number of Trips <1', 'Number of Trips 1-3', 'Number of Trips 3-5', 'Number of Trips 5-10',
                'Number of Trips 10-25', 'Number of Trips 25-50', 'Number of Trips 50-100']

# Using joblib's Parallel and delayed to compute in parallel
averages = Parallel(n_jobs=-1)(delayed(calculate_average_trips)(trips_by_distance, col_name) for col_name in column_names)

for col_name, average in zip(column_names, averages):
    print(f"Average {col_name}: {average}")

