# Machine Learning with Python
This is step by step tutorial for Machine Learning using Python.

## Data PreProcessing | Day 1
Check out the code from [here] 
### Step 1 - Importing the required Libraries
  There are two essential libraries which we will immport every time.
  * NumPy - NumPy is a library which contains Mathematical functions.
  * Pandas - Pandas is the library used to import and manage the datasets.
### Step 2 - Importing the Data set
  Data sets are generally available in .csv format. A CSV file stores tabular data in plain text. Each line of the file is a data record.   We use 'read_csv' method of the pandas library to read a local CSV file as dataframe. Then we make seperate Matrix and Vector of the independent and dependent variables from the dataframe.
### Step 3 - Handling the Missing Data
  The data we get is rarely homogeneous. Data can be missing due to various reasons and needs to be handled so that it does not reduce the performance of our machine learning model. We can replace the missing data by the Mean or Median of the entire column. We use Imputer class of sklearn.preprocessing for this task. 

### Step 4 - Encoding Categorical Data
  Categorical data are variables that contain label values rather than numeric values. The number of possible values are often limited to a fixed set. Example values such as 'Yes' and 'No' cannot be used in mathematical equations of the model so we need to encode these variable into numbers. To achieve this we import LabelEncoder class from sklearn.preprocessing library.
  
### Step 5 - Splittling the dataset into test set and training set
  We make two partitions of dataset one for training the model called training set and other for testing the performance of the trained model called test set. The split is generally 80/20 but it may varies in some cases. We import train_test_split() method of sklearn.crossvalidation library.

### Step 6 - Feature Scaling 
  Most of the machine learning algorithms use the Euclidean distance between two points in their computations, features highly varying in magnitudes, units and range pose problems. High magnitudes features will weigh more in the distance calculations than features with low magnitudes. Done by Feature standarzation or Z-score nomalization. StandardScalar of sklearn.preprocessing is imported.
