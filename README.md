# dsl
## Experiment 1: Working with NumPy Arrays
### code
```
import numpy as np
import pandas as pd

# 1. Load dataset from a URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)  # Read CSV into a DataFrame

# 2. Convert specific columns to a NumPy array
data_np = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

# 3. Indexing, Slicing, Reshaping
sliced_arr = data_np[:5, :2]  # First 5 rows, first 2 columns
reshaped_arr = data_np[:10].reshape(5, 4)  # Reshape first 10 rows into (5,4)

# 4. Basic Arithmetic Operations
sum_array = data_np[:5] + 1
mul_array = data_np[:5] * 2

# 5. Additional Operations
stacked_v = np.vstack((data_np[:3], data_np[3:6]))  # Vertical stacking
stacked_h = np.hstack((data_np[:3], data_np[3:6]))  # Horizontal stacking

# 6. Statistical and Sorting Operations
min_value = np.min(data_np, axis=0)
max_value = np.max(data_np, axis=0)
total_sum = np.sum(data_np, axis=0)
sorted_arr = np.sort(data_np[:10], axis=0)

# 7. Identity Matrix
identity_matrix = np.eye(4)  # 4x4 identity matrix

# 8. Conditional Replacement using where()
modified_arr = np.where(data_np[:5] > 5, 1, 0)

# Display results
print("First 5 rows of dataset:\n", data_np[:5])
print("Sliced Array:\n", sliced_arr)
print("Reshaped Array:\n", reshaped_arr)
print("Array + 1:\n", sum_array)
print("Element-wise Multiplication:\n", mul_array)
print("Vertically Stacked Arrays:\n", stacked_v)
print("Horizontally Stacked Arrays:\n", stacked_h)
print("Min Values:", min_value, "Max Values:", max_value, "Sum:", total_sum)
print("Sorted Array:\n", sorted_arr)
print("Identity Matrix:\n", identity_matrix)
print("Modified Array with Condition:\n", modified_arr)
```
### sample output 
```
First 5 rows of dataset:
 [[5.1 3.5 1.4 0.2]
  [4.9 3.0 1.4 0.2]
  [4.7 3.2 1.3 0.2]
  [4.6 3.1 1.5 0.2]
  [5.0 3.6 1.4 0.2]]
Sliced Array:
 [[5.1 3.5]
  [4.9 3. ]
  [4.7 3.2]
  [4.6 3.1]
  [5.  3.6]]
Reshaped Array:
 [[5.1 3.5 1.4 0.2]
  [4.9 3.  1.4 0.2]
  [4.7 3.2 1.3 0.2]
  [4.6 3.1 1.5 0.2]
  [5.  3.6 1.4 0.2]]
Array + 1:
 [[6.1 4.5 2.4 1.2]
  [5.9 4.  2.4 1.2]
  [5.7 4.2 2.3 1.2]
  [5.6 4.1 2.5 1.2]
  [6.  4.6 2.4 1.2]]
Element-wise Multiplication:
 [[10.2  7.  2.8 0.4]
  [ 9.8  6.  2.8 0.4]
  [ 9.4  6.4 2.6 0.4]
  [ 9.2  6.2 3.  0.4]
  [10.   7.2 2.8 0.4]]
Vertically Stacked Arrays:
 [[5.1 3.5 1.4 0.2]
  [4.9 3.  1.4 0.2]
  [4.7 3.2 1.3 0.2]
  [4.6 3.1 1.5 0.2]
  [5.  3.6 1.4 0.2]
  [5.4 3.9 1.7 0.4]]
Horizontally Stacked Arrays:
 [[5.1 3.5 1.4 0.2 4.6 3.1 1.5 0.2]
  [4.9 3.  1.4 0.2 5.  3.6 1.4 0.2]
  [4.7 3.2 1.3 0.2 5.4 3.9 1.7 0.4]]
Min Values: [4.3 2.  1.  0.1] Max Values: [7.9 4.4 6.9 2.5] Sum: [876.5 458.6 563.7 179.9]
Sorted Array:
 [[4.3 2.  1.  0.1]
  [4.4 2.9 1.2 0.2]
  [4.6 3.  1.3 0.2]
  [4.7 3.1 1.3 0.2]
  [4.8 3.2 1.4 0.2]]
Identity Matrix:
 [[1. 0. 0. 0.]
  [0. 1. 0. 0.]
  [0. 0. 1. 0.]
  [0. 0. 0. 1.]]
Modified Array with Condition:
 [[1 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [1 0 0 0]]
```
### Experiment 2: Working with Pandas DataFrame
### code 
```

import pandas as pd

# 1. Load dataset from an online source
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Display basic details
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n", df.info())

# 3. Handling missing values
df.fillna({'Age': df['Age'].mean()}, inplace=True)  # Replace NaN in Age with mean
df.dropna(subset=['Embarked'], inplace=True)  # Drop rows with missing Embarked

# 4. Filtering and sorting
filtered_df = df[df['Age'] > 30]  # Passengers older than 30
sorted_df = df.sort_values(by="Fare", ascending=False)  # Sort by Fare

# 5. Grouping and aggregation
grouped_df = df.groupby("Pclass")["Survived"].mean()  # Survival rate by class

# 6. Adding a new column
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # Family size calculation

# 7. Statistical operations
average_fare = df["Fare"].mean()
max_age = df["Age"].max()

# Display results
print("\nFiltered Data (Age > 30):\n", filtered_df.head())
print("\nSorted Data (Highest Fare First):\n", sorted_df.head())
print("\nSurvival Rate by Class:\n", grouped_df)
print("\nAverage Fare:", average_fare, "Max Age:", max_age)
print("\nUpdated DataFrame with Family Size:\n", df.head())
```
### sample output 
```
First 5 rows:
    PassengerId  Survived  Pclass  ... Cabin Embarked
0            1         0       3  ...   NaN        S
1            2         1       1  ...   C85        C
2            3         1       3  ...   NaN        S
3            4         1       1  ...  C123        S
4            5         0       3  ...   NaN        S

Dataset Info:
 RangeIndex: 891 entries, 0 to 890
 Data columns (total 12 columns):
  #   Column        Non-Null Count  Dtype  
 ---  ------        --------------  ------
  0   PassengerId  891 non-null    int64  
  1   Survived     891 non-null    int64  
  2   Pclass       891 non-null    int64  
  3   Name         891 non-null    object 
  4   Sex          891 non-null    object 
  5   Age          714 non-null    float64
  6   SibSp        891 non-null    int64  
  7   Parch        891 non-null    int64  
  8   Ticket       891 non-null    object 
  9   Fare         891 non-null    float64
 10   Cabin        204 non-null    object 
 11   Embarked     889 non-null    object 

Filtered Data (Age > 30):
    PassengerId  Survived  Pclass  ... Cabin Embarked
1            2         1       1  ...   C85        C
6            7         0       1  ...   E46        S
11          12         1       1  ...   NaN        S
15          16         1       2  ...   NaN        S
33          34         0       2  ...   NaN        S

Sorted Data (Highest Fare First):
    PassengerId  Survived  Pclass  ... Cabin Embarked
258        259         1       1  ...  B51 B53 B55  C
679        680         1       1  ...          C
737        738         1       1  ...          C
27          28         0       1  ...          C
88          89         1       1  ...  B42  S

Survival Rate by Class:
 Pclass
1    0.629630
2    0.472826
3    0.242363
Name: Survived, dtype: float64

Average Fare: 32.2042079685746 Max Age: 80.0

Updated DataFrame with Family Size:
    PassengerId  Survived  Pclass  ... Cabin Embarked  FamilySize
0            1         0       3  ...   NaN        S          2
1            2         1       1  ...   C85        C          2
2            3         1       3  ...   NaN        S          1
3            4         1       1  ...  C123        S          2
4            5         0       3  ...   NaN        S          1
```
