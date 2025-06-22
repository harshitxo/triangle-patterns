# Importing libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Print first 5 rows
print("First few rows of the dataset:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dropping rows with missing 'age' or 'embarked' (small number of rows only)
df.dropna(subset=['age', 'embarked'], inplace=True)

sns.countplot(x='survived', data=df)
plt.title("Total Survival Count (0 = Died, 1 = Survived)")
plt.show()

sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.countplot(x='class', hue='survived', data=df)
plt.title("Survival Count by Class")
plt.show()

sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.show()

sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare (Color indicates survival)")
plt.show()

