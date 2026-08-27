# ML-labsheet-2

sample_data.csv file 
Name,Age,Salary,City,Join_Date
Aman,21,25000,Delhi,2023-01-15
Riya,22,30000,Roorkee,2022-06-20
Rahul,,28000,Haridwar,2021-09-10
Priya,23,,Dehradun,2023-03-25
Karan,22,100000,Delhi,2020-12-05

Q1

import pandas as pd
data = pd.read_csv("sample_data.csv")
missing_values = data.isnull()
print("Missing Values:")
print(missing_values)

Q2

import pandas as pd
data = pd.read_csv("sample_data.csv")
missing_percentage=data.isnull().mean() * 100
print("Percentage of Missing Values:")
print(missing_percentage)

Q3

import pandas as pd
data = pd.read_csv("sample_data.csv")
cleaned_data = data.dropna()
print("Dataset after removing missing rows:")
print(cleaned_data)

Q4

import pandas as pd
data = pd.read_csv("sample_data.csv")
missing_percentage = data.isnull().mean()
cleaned_data = data.loc[:, missing_percentage <= 0.50]
print("Dataset after removing columns with more than 50% missing values:")
print(cleaned_data)

Q5

import pandas as pd
data = pd.read_csv("sample_data.csv")
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Salary"] = data["Salary"].fillna(data["Salary"].mean())
print("Dataset after mean imputation:")
print(data)

Q6

import pandas as pd
data = pd.read_csv("sample_data.csv")
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Salary"] = data["Salary"].fillna(data["Salary"].median())
print("Dataset after median imputation:")
print(data)

Q7

import pandas as pd
data = pd.read_csv("sample_data.csv")
data["City"] = data["City"].fillna(data["City"].mode()[0])
print("Dataset after mode imputation:")
print(data)

Q8

import pandas as pd
data = pd.read_csv("sample_data.csv")
data = data.ffill()
print("Dataset after forward fill:")
print(data)

Q9

import pandas as pd
data = pd.read_csv("sample_data.csv")
data = data.bfill()
print("Dataset after backward fill:")
print(data)

Q10

import pandas as pd
original_data = pd.read_csv("sample_data.csv")
processed_data = original_data.copy()
processed_data["Age"] = processed_data["Age"].fillna(
    processed_data["Age"].median()
)
processed_data["Salary"] = processed_data["Salary"].fillna(
    processed_data["Salary"].median()
)
print("Original Dataset:")
print(original_data)
print("\nProcessed Dataset:")
print(processed_data)
print("\nMissing values before processing:")
print(original_data.isnull().sum())
print("\nMissing values after processing:")
print(processed_data.isnull().sum())

Q11

import pandas as pd
data = pd.read_csv("sample_data.csv")

# Calculate Q1 and Q3
Q1 = data["Salary"].quantile(0.25)
Q3 = data["Salary"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

#define limits 
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers = data[
    (data["Salary"] < lower_limit) |
    (data["Salary"] > upper_limit)
]
print("Outliers:")
print(outliers)

Q12

import pandas as pd
from scipy.stats import zscore
data = pd.read_csv("sample_data.csv")
data["Z_Score"] = zscore(data["Salary"])
outliers = data[abs(data["Z_Score"]) > 3]
print("Outliers using Z-score:")
print(outliers)

Q13

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("sample_data.csv")
sns.boxplot(y=data["Salary"])
plt.title("Salary Box Plot")
plt.ylabel("Salary")
plt.show()

Q14

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("sample_data.csv")
plt.scatter(data["Age"], data["Salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

Q15

import pandas as pd
data = pd.read_csv("sample_data.csv")

Q1 = data["Salary"].quantile(0.25)
Q3 = data["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
cleaned_data = data[
    (data["Salary"] >= lower_limit) &
    (data["Salary"] <= upper_limit)
]
print("Dataset after removing outliers:")
print(cleaned_data)

Q16

import pandas as pd
data = pd.read_csv("sample_data.csv")

Q1 = data["Salary"].quantile(0.25)
Q3 = data["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

median_salary = data["Salary"].median()
data["Salary"] = data["Salary"].apply(
    lambda x: median_salary
    if x < lower_limit or x > upper_limit
    else x
)
print("Dataset after replacing outliers:")
print(data)

Q17

import pandas as pd
data = pd.read_csv("sample_data.csv")
lower_limit = data["Salary"].quantile(0.05)
upper_limit = data["Salary"].quantile(0.95)
data["Salary"] = data["Salary"].clip(
    lower_limit,
    upper_limit
)
print("Dataset after percentile capping:")
print(data)

Q18

import pandas as pd

original_data = pd.read_csv("sample_data.csv")

# Create copy
processed_data = original_data.copy()

Q1 = processed_data["Salary"].quantile(0.25)
Q3 = processed_data["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers
processed_data = processed_data[
    (processed_data["Salary"] >= lower_limit) &
    (processed_data["Salary"] <= upper_limit)
]
print("Original Dataset:")
print(original_data)
print("\nDataset after Outlier Treatment:")
print(processed_data)

Q19

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("sample_data.csv")

# Select numerical features
features = data[["Age", "Salary"]].copy()

# Apply Min-Max scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(features)

normalized_data = pd.DataFrame(
    normalized_data,
    columns=["Age", "Salary"]
)
print("Min-Max Normalized Data:")
print(normalized_data)

Q20

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("sample_data.csv")

features = data[["Age", "Salary"]].copy()

# Apply Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(features)

standardized_data = pd.DataFrame(
    standardized_data,
    columns=["Age", "Salary"]
)
print("Standardized Data:")
print(standardized_data)

Q21

import pandas as pd
from sklearn.preprocessing import RobustScaler

data = pd.read_csv("sample_data.csv")

features = data[["Age", "Salary"]].copy()

# Apply Robust Scaling
scaler = RobustScaler()
robust_data = scaler.fit_transform(features)

robust_data = pd.DataFrame(
    robust_data,
    columns=["Age", "Salary"]
)
print("Robust Scaled Data:")
print(robust_data)

Q22

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
data = pd.read_csv("sample_data.csv")
features = data[["Age", "Salary"]].copy()

# Apply Max Absolute Scaling
scaler = MaxAbsScaler()
scaled_data = scaler.fit_transform(features)

scaled_data = pd.DataFrame(
    scaled_data,
    columns=["Age", "Salary"]
)
print("Max Absolute Scaled Data:")
print(scaled_data)

Q23

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("sample_data.csv")
features = data[["Age", "Salary"]].copy()

# Normalize data
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(features)

normalized_data = pd.DataFrame(
    normalized_values,
    columns=["Age", "Salary"]
)
print("Original Data:")
print(features)
print("\nNormalized Data:")
print(normalized_data)

Q24

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("sample_data.csv")

# Select Salary
salary = data[["Salary"]].dropna()

# Normalize Salary
scaler = MinMaxScaler()
normalized_salary = scaler.fit_transform(salary)

# Plot original data
plt.hist(salary["Salary"], bins=5)
plt.title("Original Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Plot normalized data
plt.hist(normalized_salary, bins=5)
plt.title("Normalized Salary Distribution")
plt.xlabel("Normalized Salary")
plt.ylabel("Frequency")
plt.show()

Q25

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("sample_data.csv")
salary = data[["Salary"]].dropna()

# Standardize salary
scaler = StandardScaler()
scaled_salary = scaler.fit_transform(salary)

# Original box plot
plt.boxplot(salary["Salary"])
plt.title("Original Salary")
plt.show()

# Scaled box plot
plt.boxplot(scaled_salary)
plt.title("Standardized Salary")
plt.show()

Q26

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler
)
data = pd.read_csv("sample_data.csv")
features = data[["Age", "Salary"]].dropna()

# Apply different scaling techniques
minmax = MinMaxScaler().fit_transform(features)
standard = StandardScaler().fit_transform(features)
robust = RobustScaler().fit_transform(features)
maxabs = MaxAbsScaler().fit_transform(features)
print("Original Data:")
print(features)
print("\nMin-Max Scaling:")
print(minmax)
print("\nStandardization:")
print(standard)
print("\nRobust Scaling:")
print(robust)
print("\nMax Absolute Scaling:")
print(maxabs)

Q27

import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("sample_data.csv")

# Create label encoder
encoder = LabelEncoder()

# Encode City column
data["City_Encoded"] = encoder.fit_transform(data["City"])
print(data)

Q28

import pandas as pd
data = pd.read_csv("sample_data.csv")

# Apply one-hot encoding
encoded_data = pd.get_dummies(
    data,
    columns=["City"]
)
print(encoded_data)

Q29

pip install category_encoders
import pandas as pd
import category_encoders as ce
data = pd.read_csv("sample_data.csv")

# Create binary encoder
encoder = ce.BinaryEncoder(cols=["City"])

# Apply binary encoding
encoded_data = encoder.fit_transform(data)
print(encoded_data)

Q30

import pandas as pd
data = pd.read_csv("sample_data.csv")

# Combine two columns
data["Student_Location"] = (
    data["Name"] + "_" + data["City"]
)
print(data)

Q31

import pandas as pd
data = pd.read_csv("sample_data.csv")

# Convert Join_Date to datetime
data["Join_Date"] = pd.to_datetime(data["Join_Date"])

# Extract date components
data["Year"] = data["Join_Date"].dt.year
data["Month"] = data["Join_Date"].dt.month
data["Day"] = data["Join_Date"].dt.day
print(data)

Q32

import pandas as pd
data = pd.read_csv("sample_data.csv")

# Mathematical transformation
data["Salary_in_Thousands"] = data["Salary"] / 1000
print(data)

Q33

import pandas as pd
import numpy as np
data = pd.read_csv("sample_data.csv")

# Apply log transformation
data["Log_Salary"] = np.log1p(data["Salary"])
print(data[["Salary", "Log_Salary"]])

Q34

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("sample_data.csv")

# Select numerical columns
numerical_data = data.select_dtypes(
    include="number"
)

# Calculate correlation
correlation_matrix = numerical_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)
plt.title("Feature Correlation")
plt.show()

Q35

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load original dataset
data = pd.read_csv("sample_data.csv")

# Create a copy
processed_data = data.copy()

# Handle missing numerical values
processed_data["Age"] = processed_data["Age"].fillna(
    processed_data["Age"].median()
)
processed_data["Salary"] = processed_data["Salary"].fillna(
    processed_data["Salary"].median()
)

# Convert date column
processed_data["Join_Date"] = pd.to_datetime(
    processed_data["Join_Date"]
)

# Extract date features
processed_data["Year"] = processed_data["Join_Date"].dt.year
processed_data["Month"] = processed_data["Join_Date"].dt.month
processed_data["Day"] = processed_data["Join_Date"].dt.day

# One-hot encode City
processed_data = pd.get_dummies(
    processed_data,
    columns=["City"]
)

# Remove original date column
processed_data = processed_data.drop(
    "Join_Date",
    axis=1
)

# Select numerical columns for scaling
numerical_columns = [
    "Age",
    "Salary",
    "Year",
    "Month",
    "Day"
]

# Apply Min-Max Scaling
scaler = MinMaxScaler()
processed_data[numerical_columns] = scaler.fit_transform(
    processed_data[numerical_columns]
)

# Save final dataset
processed_data.to_csv(
    "final_preprocessed_dataset.csv",
    index=False
)
print("Final Preprocessed Dataset:")
print(processed_data)
print("\nFinal dataset saved successfully.")