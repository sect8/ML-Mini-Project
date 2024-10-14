import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading the dataset...")
df = pd.read_csv('D:/ML MINI PROJECT/Weather_Prediction/seattle-weather.csv')

print("\nDisplaying basic information about the dataset:")
print(df.info())

print("\nShowing the first few rows of the dataset:")
print(df.head())

print("\nDisplaying summary statistics of the dataset:")
print(df.describe())

print("\nChecking for missing values in each column:")
print(df.isnull().sum())

print("\nCreating histograms for numerical columns...")
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

print("\nDisplaying the frequency of different weather conditions:")
print(df['weather'].value_counts())

print("\nShowing the correlation matrix between numerical variables:")
print(df.corr())

print("\nConverting 'date' column to datetime and setting it as index...")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("\nPlotting maximum temperature over time...")
df['temp_max'].plot(figsize=(12,6))
plt.title('Maximum Temperature Over Time')
plt.ylabel('Temperature (°C)')
plt.show()

print("\nCalculating and plotting monthly average maximum temperature...")
df['month'] = df.index.month
monthly_avg = df.groupby('month').mean()
monthly_avg['temp_max'].plot(kind='bar', figsize=(10,6))
plt.title('Average Maximum Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.show()

print("\nCreating a scatter plot of maximum temperature vs precipitation...")
sns.scatterplot(data=df, x='temp_max', y='precipitation')
plt.title('Maximum Temperature vs Precipitation')
plt.xlabel('Maximum Temperature (°C)')
plt.ylabel('Precipitation (mm)')
plt.show()
