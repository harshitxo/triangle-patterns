import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

df = sns.load_dataset('titanic')
df.head()

# Basic Info
df.info()

# Statistical Summary
df.describe(include='all')

# Shape of the dataset
df.shape

# Check for missing values
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing, 'Percent': missing_percent})
missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

df.dtypes
df.nunique()

sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.xticks([0, 1], ['No', 'Yes'])

sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title("Age Distribution")

sns.boxplot(x=df['fare'])
plt.title("Fare Boxplot")

sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survival by Gender")

sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival by Passenger Class")

sns.kdeplot(data=df[df['survived']==1]['age'].dropna(), label='Survived', shade=True)
sns.kdeplot(data=df[df['survived']==0]['age'].dropna(), label='Not Survived', shade=True)
plt.title("Age Distribution by Survival")
plt.legend()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

sns.countplot(x='embark_town', hue='survived', data=df)
plt.title("Survival by Embarkation Town")

