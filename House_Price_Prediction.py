import pandas as pd
import numpy as np

#Plot
import matplotlib.pyplot as plt
import seaborn as sns

#Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

# x-y split
from sklearn.model_selection import train_test_split

#Scaling
from sklearn.preprocessing import StandardScaler

# Regression algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor  # Multi-layer perceptron regressor (MLP)
from sklearn.impute import KNNImputer

#MSE AND MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

#
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()

train.info()

train.shape

train.columns

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap', fontsize=16)
plt.xlabel('Columns')
plt.show()

missing_values = train.isnull().sum() 
missing_values = (missing_values[missing_values > 0] / len(train) ) * 100
print(missing_values.sort_values(ascending=False))

train[train['PoolQC'].isna()]['PoolArea'].unique()
sns.scatterplot(data=train,x='PoolArea',y='SalePrice')

train['PoolQC'].value_counts()
train.drop(columns=['PoolQC','PoolArea'],inplace=True)
train['MiscFeature'].unique()

plt.figure(figsize=(10, 4))

# Plot 1: Violin plot (shows distribution shape + density)
ax1 = plt.subplot(121)
sns.violinplot(data=train, x='MiscFeature', y='SalePrice', inner='stick')
plt.title('Price Distribution by MiscFeature')
plt.xticks(rotation=45)

# Plot 2: Countplot with ANNOTATIONS (show sample sizes)
ax2 = plt.subplot(122)
sns.countplot(data=train, x='MiscFeature')
plt.title('Count of MiscFeature Categories')

# Annotate each bar with exact counts
for p in ax2.patches:
    ax2.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom')

plt.tight_layout()
plt.show()

train.drop(columns=['MiscFeature'],inplace=True)

train['Street'].value_counts()
train['Alley'].value_counts()
train.drop(columns=['Street','Alley'],inplace=True)
train['Fence'].value_counts()
plt.title('Price Distribution by Fence')
sns.violinplot(data=train,x='Fence',y='SalePrice')

train.drop(columns=['Fence'],inplace=True)
plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.title('Count Distribution by MasVnrType')
sns.countplot(data=train,x='MasVnrType')
plt.subplot(1,2,2)
plt.title('Price Distribution by MasVnrType')
sns.violinplot(data=train,x='MasVnrType',y='SalePrice')
plt.show()

train[train['MasVnrType'].isnull() & (train['MasVnrArea'] > 0)]

train.loc[(train['MasVnrType'].isnull()) & (train['MasVnrArea'] == 0), 'MasVnrType'] = 'None'
train['MasVnrType'] = train['MasVnrType'].fillna('Unknown')
train[train['FireplaceQu'].isna()]['Fireplaces'].unique()

plt.figure(figsize=(14,7))
plt.subplot(1,2,1)
plt.title('Count Distribution by FireplaceQu')
sns.countplot(data=train,x='FireplaceQu')
plt.subplot(1,2,2)
plt.title('Price Distribution by FireplaceQu')
sns.violinplot(data=train,x='FireplaceQu',y='SalePrice')
plt.show()

train["FireplaceQu"].fillna('None', inplace=True)

plt.figure(figsize=(12,6))
train.corr(numeric_only=True)['LotFrontage'].sort_values().drop('LotFrontage').plot(kind='bar')

features = ['1stFlrSF', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 'TotRmsAbvGrd', 'GarageArea']
lotfrontage_df = train[features + ['LotFrontage']]

known = lotfrontage_df[lotfrontage_df['LotFrontage'].notnull()]
unknown = lotfrontage_df[lotfrontage_df['LotFrontage'].isnull()]

X_known = known[features]
y_known = known['LotFrontage']
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_known, y_known)

X_unknown = unknown[features]
predicted = reg.predict(X_unknown)

train.loc[train['LotFrontage'].isnull(), 'LotFrontage'] = predicted

sns.distplot(train['LotFrontage'])

garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train[garage_cols] = train[garage_cols].fillna('None')
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['YearBuilt'])
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
train[bsmt_cols] = train[bsmt_cols].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

plt.figure(figsize=(10,6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap', fontsize=16)
plt.xlabel('Columns')
plt.show()

duplicates_values = train.duplicated()
duplicates_values = (duplicates_values[duplicates_values > 0] / len(train) ) * 100
print(duplicates_values)

train.drop(columns=['Id'],inplace=True)

IMBALANCE_THRESHOLD = 0.95
target_variable = 'SalePrice'

# Select categorical columns
categorical_cols = train.select_dtypes(include='object').columns

# To collect columns dropped due to imbalance
dropped_columns = []

for col in categorical_cols:
    value_counts = train[col].value_counts(normalize=True)
    top_ratio = value_counts.iloc[0]

    print(f"Analyzing column: {col}")
    print(value_counts)

    if top_ratio >= IMBALANCE_THRESHOLD:
        print(f"🔻 Dropping '{col}' due to imbalance (top category = {value_counts.index[0]}, ratio = {top_ratio:.2f})\n")
        train.drop(col, axis=1, inplace=True)
        dropped_columns.append(col)
    else:
        plt.figure(figsize=(16, 6))
        # Violinplot
        plt.subplot(1,2,1)
        order = train.groupby(col)[target_variable].median().sort_values().index
        sns.violinplot(x=train[col], y=train[target_variable],order=order)
        plt.xticks(rotation=45)
        plt.title(f'Violinplot - {col} vs {target_variable}')

        # Pointplot
        plt.subplot(1,2,2)
        # Try to preserve category order if it makes sense
        ordered_cats = train.groupby(col)[target_variable].mean().sort_values().index.tolist()
        sns.pointplot(x=col, y=target_variable, data=train, order=ordered_cats)
        plt.xticks(rotation=45)
        plt.title(f'Pointplot - {col} vs {target_variable}')
        plt.tight_layout()
        plt.show()

print("✅ Dropped columns due to imbalance:", dropped_columns)

relevant_categorical_features = [
    'Neighborhood', 'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish', 
    'Foundation', 'MSZoning', 'FireplaceQu', 'HeatingQC', 'MasVnrType',
    'Exterior1st', 'Exterior2nd', 'HouseStyle', 'BsmtExposure', 'GarageType',
    'SaleCondition', 'RoofStyle', 'CentralAir', 'LotShape', 'Condition1'
]
ambig_categorical_features = [
    "LotShape", "LandContour", "LotConfig", "Condition1", "BldgType","SaleType"
]

desc_df = train[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
       'SalePrice']].describe().T
desc_df

# Plot as heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(desc_df[['count','mean', 'std', 'min', '25%', '50%', '75%', 'max']], 
            annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Heatmap of Summary Statistics for Numerical Features')
plt.tight_layout()
plt.show()

# Columns of interest
cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalBsmtSF', 
        'GrLivArea']

# Set plot style
sns.set(style="whitegrid")

# Plot histograms
def plot_histograms(df, columns):
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(columns):
        plt.subplot(4, 2, i + 1)
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.suptitle("Histograms of Selected Features", fontsize=16, y=1.02)
    plt.show()

# Plot boxplots
def plot_boxplots(df, columns):
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(columns):
        plt.subplot(4, 2, i + 1)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.suptitle("Boxplots of Selected Features", fontsize=16, y=1.02)
    plt.show()

# Run visualizations
plot_histograms(train, cols)
plot_boxplots(train, cols)

def cap_outliers_iqr(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df[column] = df[column].clip(lower, upper)

# Columns to treat
cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalBsmtSF', 
        'GrLivArea']

for col in cols:
    cap_outliers_iqr(train, col)

# Columns of interest
cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalBsmtSF', 
        'GrLivArea','SalePrice']

# Set plot style
sns.set(style="whitegrid")

# Plot histograms
def plot_histograms(df, columns):
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(columns):
        plt.subplot(4, 2, i + 1)
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.suptitle("Histograms of Selected Features", fontsize=16, y=1.02)
    plt.show()

# Plot boxplots
def plot_boxplots(df, columns):
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(columns):
        plt.subplot(4, 2, i + 1)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.suptitle("Boxplots of Selected Features", fontsize=16, y=1.02)
    plt.show()

# Run visualizations
plot_histograms(train, cols)
plot_boxplots(train, cols)

plt.figure(figsize=(12,6))
sns.heatmap(train[['BsmtFinSF2', '3SsnPorch', 'ScreenPorch', 'BsmtHalfBath', 'EnclosedPorch','SalePrice']].corr(numeric_only=True), annot=True,cbar=True, cmap='viridis')
plt.title('Correlation Matrix between sparse variables and Sale Price')

cols_to_drop_corr = [
    'BsmtFinSF2',
    '3SsnPorch',
    'ScreenPorch',
    'BsmtHalfBath',
    'EnclosedPorch'
]

train.drop(columns=cols_to_drop_corr, inplace=True)
print(f"{len(cols_to_drop_corr)} low-correlation columns dropped. New shape: {train.shape}")

plt.figure(figsize=(30,12))
sns.heatmap(data=train.corr(numeric_only=True),annot=True,cmap='viridis')
plt.title('Correlation Matrix between variables and Sale Price', fontsize=20)

plt.figure(figsize=(12,6))
train.corr(numeric_only=True)['SalePrice'].sort_values().drop('SalePrice').plot(kind='bar')
plt.suptitle('Correlation between variables and Sale Price', fontsize=16)

for col in ['LowQualFinSF', 'MiscVal','BsmtUnfSF']:
    print(train[col].value_counts())

train.drop(columns=['OverallCond','KitchenAbvGr' , 'MSSubClass', 'MoSold', 'LowQualFinSF', 'MiscVal','BsmtUnfSF'], inplace=True)

numerical_features = train.select_dtypes(include=['int64','float64']).columns.tolist()
numerical_features

import scipy.stats as stats
import pandas as pd

# Your variables of interest
ambig_vars = ['LotShape', 'LandContour', 'LotConfig', 'Condition1', 'BldgType', 'SaleType']

# Loop over each variable
for var in ambig_vars:
    try:
        # Group SalePrice by category of the variable
        grouped_data = train.groupby(var)['SalePrice'].apply(list)
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*grouped_data)

        print(f"ANOVA for {var}: F-statistic = {f_stat:.2f}, p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ➤ Significant difference detected (p < 0.05)\n")
        else:
            print(f"  ➤ No significant difference (p ≥ 0.05)\n")

    except Exception as e:
        print(f"⚠️ Could not compute ANOVA for {var} due to error: {e}\n")

selected_cols = [
    'LotFrontage', 'LotArea',
    '1stFlrSF', '2ndFlrSF', 'GrLivArea',
    'TotalBsmtSF', 'BsmtFinSF1',
    'BsmtFullBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'FireplaceQu',
    'GarageCars', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'GarageType',
    'WoodDeckSF', 'OpenPorchSF', 'CentralAir',
    'YearBuilt', 'YearRemodAdd', 'YrSold',
    'OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual',
    'BsmtExposure', 'Foundation', 'HeatingQC',
    'HouseStyle', 'RoofStyle', 'BldgType',
    'Neighborhood', 'MSZoning', 'Condition1', 'LandContour',
    'LotShape', 'LotConfig',
    'MasVnrArea', 'MasVnrType', 'Exterior1st', 'Exterior2nd',
    'SaleCondition', 'SaleType',
    'SalePrice'
]     

final_train = train[selected_cols]
print(len(final_train.columns),final_train.columns)

plt.figure(figsize=(30,12))
sns.heatmap(data=final_train.corr(numeric_only=True),annot=True,cmap='viridis')
plt.title('Correlation Matrix between variables and Sale Price', fontsize=20)

# Create new engineered features
final_train['TotalSF'] = final_train['1stFlrSF'] + final_train['2ndFlrSF'] + final_train['TotalBsmtSF']
final_train['TotalBath'] = (final_train['FullBath'] + 0.5 * final_train['HalfBath'] +
                   final_train.get('BsmtFullBath', 0) + 0.5 * final_train.get('BsmtHalfBath', 0))
final_train['AgeAtSale'] = final_train['YrSold'] - final_train['YearBuilt']
final_train['GarageAge'] = final_train['YrSold'] - final_train['GarageYrBlt']
final_train['HasRemodeled'] = (final_train['YearBuilt'] != final_train['YearRemodAdd']).astype(int)

# Combine condition or exterior if needed
final_train['ExteriorCombo'] = final_train['Exterior1st'] + '_' + final_train['Exterior2nd']

# Drop redundant or less informative features
features_to_drop = [
    '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd',
    'GarageArea', 'GarageYrBlt', 'YearRemodAdd',
    'Exterior1st', 'Exterior2nd'
]

final_train.drop(columns=features_to_drop,inplace=True)
final_column_features = final_train.columns
final_column_features
X = final_train.drop("SalePrice", axis=1)
y = final_train["SalePrice"]
X.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
x_train.head()
# Identify object (categorical) columns in x_train
object_cols = x_train.select_dtypes(include="object").columns

# Limit to first 34 object columns (or define manually if needed)
selected_ordinal_cols = object_cols[:34]

# One-hot encode these columns in x_train
x_train_dummies = pd.get_dummies(x_train[selected_ordinal_cols], drop_first=True)

# One-hot encode same columns in x_test (use same categories as train)
x_test_dummies = pd.get_dummies(x_test[selected_ordinal_cols], drop_first=True)

# Align the train and test sets to have the same columns
x_train_dummies, x_test_dummies = x_train_dummies.align(x_test_dummies, join='left', axis=1, fill_value=0)

# Drop original ordinal columns from x_train and x_test
x_train = x_train.drop(columns=selected_ordinal_cols)
x_test = x_test.drop(columns=selected_ordinal_cols)

# Concatenate dummies to the rest of the features
x_train = pd.concat([x_train, x_train_dummies], axis=1)
x_test = pd.concat([x_test, x_test_dummies], axis=1)
y_train
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# List of linear regression models to apply
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Regressor": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(max_iter=1000)
}
# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2
# Dictionary to store the results
results = {}
for name, model in models.items(): # When you call items() on a dictionary, returns a list of the dictionary’s key-value tuple pairs.
                                   # Here "name" represents the "key", and "model" represents the "value"
  if name in ["Support Vector Regressor", "Neural Network"]: # Standardised value is only applied to support vector regressor and Neural network
        # Apply scaling for models that need it
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
  else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Compute evaluation metrics
  mae, rmse, r2 = evaluate_model(y_test, y_pred)
  results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}
# Convert results to a DataFrame for better visualization
results_train= pd.DataFrame(results).T
print(results_train)
model = Ridge()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy=r2_score(y_test,y_pred)
print(accuracy)
import pickle
with open('house_price_prediction.pickle','wb') as f:
    pickle.dump(model,f)
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test.drop(columns=['PoolQC','PoolArea'],inplace=True)
test.drop(columns=['MiscFeature'],inplace=True)
test.drop(columns=['Street','Alley'],inplace=True)
test.drop(columns=['Fence'],inplace=True)
test.loc[(test['MasVnrType'].isnull()) & (test['MasVnrArea'] == 0), 'MasVnrType'] = 'None'
test['MasVnrType'] = test['MasVnrType'].fillna('Unknown')
test["FireplaceQu"].fillna('None', inplace=True)
garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
test[garage_cols] = test[garage_cols].fillna('None')
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['YearBuilt'])
bsmt_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
test[bsmt_cols] = test[bsmt_cols].fillna('None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test.drop(columns=['OverallCond','KitchenAbvGr' , 'MSSubClass', 'MoSold', 'LowQualFinSF', 'MiscVal','BsmtUnfSF'], inplace=True)
# Create new engineered features
test['TotalSF'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']
test['TotalBath'] = (test['FullBath'] + 0.5 * test['HalfBath'] +
                   test.get('BsmtFullBath', 0) + 0.5 * test.get('BsmtHalfBath', 0))
test['AgeAtSale'] = test['YrSold'] - test['YearBuilt']
test['GarageAge'] = test['YrSold'] - test['GarageYrBlt']
test['HasRemodeled'] = (test['YearBuilt'] != test['YearRemodAdd']).astype(int)

# Combine condition or exterior if needed
test['ExteriorCombo'] = test['Exterior1st'] + '_' + test['Exterior2nd']

# Drop redundant or less informative features
features_to_drop = [
    '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'TotRmsAbvGrd',
    'GarageArea', 'GarageYrBlt', 'YearRemodAdd', 'Condition2',
    'Exterior1st', 'Exterior2nd'
]
test.drop(columns=features_to_drop,inplace=True)
X_test_final = test[[col for col in final_column_features if col != "SalePrice"]]

X_test_final = pd.get_dummies(X_test_final)
X_test_final = X_test_final.reindex(columns=x_train.columns, fill_value=0)
X_test_final = X_test_final.fillna(0)
y_pred = model.predict(X_test_final)

submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": y_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as `submission.csv`.")
