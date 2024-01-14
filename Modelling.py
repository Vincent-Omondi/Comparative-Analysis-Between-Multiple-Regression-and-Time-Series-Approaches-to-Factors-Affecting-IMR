#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Model Evaluation

# ### Data Setup
# #### Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # Adding Random Forest
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, roc_curve, auc, precision_score, plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


random_seed = 123
working_directory = r"C:\\Users\\Vincent\\Desktop\\Project\\Final_Year_Project"
os.chdir(working_directory)
pd.set_option('display.max_columns', None)
imr_df = pd.read_csv("IMR_dataset.csv") 
imr_df.head()  


# #### Exploring Data
# 
# - Checking the unique values in each column of the dataset.

# In[3]:


imr_df.describe()


# In[4]:


for column in imr_df.columns:
    unique_values = imr_df[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


# __Selecting a subset of features and the target variable (factors and y).__

# In[5]:


factors = ['Education_Level', 'Marital_Status', 'Wealth_Index', 'Residential_Type_Label', 'County_code', 'Religion_code', 'Sex_of_child']

# Defining X (factors) and y (target)
X = imr_df[factors]
y = imr_df['Child_Alive_or_Dead']


# ### Data Preprocessing
# 
# - Encoding categorical variables in the feature set using one-hot encoding.
# - Split the data into training and testing sets (70% training, 30% testing).

# In[6]:


# Encoding categorical variables 
X = pd.get_dummies(X, columns=['Residential_Type_Label'], drop_first=True)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# ## Handling Class Imbalance
# 
# #### Synthetic Minority Over-sampling Technique (SMOTE)
# 
# - Using SMOTE to handle class imbalance by oversampling the minority class.
# - Standardizing the feature data for consistency.

# In[7]:


sm = SMOTE(random_state=123)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


#  #### Model Evaluation Functions
#  
#  Defining utility functions for:
# - Plotting Receiver Operating Characteristic (ROC) curves.
# - Evaluating model performance and visualizing results.

# In[8]:


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[9]:


def eval_model(model, X_test, Y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    conf_mat = confusion_matrix(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    plot_confusion_matrix(model, X_test, y_test)
    plt.show()

    #print(conf_mat)
    print("\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    #Show ROC Curve 
    fpr, tpr, threshold = roc_curve(y_test, probs[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("AUC: ", roc_auc)

    plot_roc_curve(fpr, tpr, roc_auc)

    results_df = pd.DataFrame()
    results_df['true_class'] = y_test
    results_df['predicted_class'] = list(preds)
    results_df['survival_prob'] = probs[:, 1]

    #plot the distribution of probabilities for the estimated classes 
    sns.distplot(results_df[results_df['true_class'] == 0]['survival_prob'], label="Child is dead", hist=False)
    sns.distplot(results_df[results_df['true_class'] == 1]['survival_prob'], label="Child not dead", hist=False)
    plt.title('Distribution of Probabilities for Estimated Classes')
    plt.legend(loc='best')
    plt.show()
    
    #see the true class versus predicted class as a percentage
    print(results_df.groupby('true_class')['predicted_class'].value_counts(normalize=True))


# ## Logistic Regression Model
# 
# - Fit a logistic regression model to the resampled training data.
# - Predicting the target variable on the test data.
# - Evaluating the logistic regression model using the defined evaluation function.

# In[10]:


model = LogisticRegression(random_state=123)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)
eval_model(model, X_test, y_test)


# ## Random Forest Classifier with Class Weights
# 
# - Calculating class weights for handling class imbalance.
# - Standardizing features again.
# - Fitting a Random Forest Classifier with class weights.
# - Predicting the target variable on the test data using the Random Forest Classifier.
# - Evaluating the Random Forest Classifier using the evaluation function.

# In[11]:


# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)

# Standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Fit a Random Forest Classifier with class weights
rfc_model = RandomForestClassifier(class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=123)
rfc_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test data
y_pred = rfc_model.predict(X_test)

# Evaluating the model
eval_model(rfc_model, X_test, y_test)


# ## Random Forest Classifier with Custom Class Weights
# 
# - Specifying custom class weights for the Random Forest Classifier.
# - Fitting a Random Forest Classifier with custom class weights.
# - Evaluating the Random Forest Classifier with custom class weights using the evaluation function.

# In[12]:


weights = {1:0.30, 0:0.70}

rfc_model = RandomForestClassifier(n_estimators = 100, max_depth = 15, class_weight=weights, random_state=123)

rfc_model.fit(X_train, y_train)
eval_model(rfc_model, X_test, y_test)


# ## Data Downsampling and Model Evaluation
# 
# - Defining a function for balancing the training data by either upsampling or downsampling.
# - Downsampling the training data using the function.
# - Printing the class distribution after downsampling.
# - Fitting a Random Forest Classifier to the downsampled data.
# - Evaluating the model after downsampling using the evaluation function.

# In[13]:


def balance_sample(x_train, y_train, sample_mode='up'):
    train_df = x_train.copy()
    train_df['Child_Alive_or_Dead'] = y_train

    train_minority = train_df[train_df['Child_Alive_or_Dead'] == 0]
    train_majority = train_df[train_df['Child_Alive_or_Dead'] == 1]

    train_sampled_df = pd.DataFrame()

    if sample_mode == 'down':
        train_majority_down = resample(train_majority, replace=False,  n_samples=train_minority.shape[0], random_state=123)
        train_sampled_df = pd.concat([train_minority, train_majority_down])  
    else:
        train_minority_up = resample(train_minority, replace=True,  n_samples=train_majority.shape[0], random_state=123)
        train_sampled_df = pd.concat([train_majority, train_minority_up])

    x_train_samp = train_sampled_df.drop(['Child_Alive_or_Dead'], axis=1)
    y_train_samp = train_sampled_df['Child_Alive_or_Dead']

    return x_train_samp, y_train_samp 


# In[14]:


#downsample random forest
X_train_dwn, y_train_dwn = balance_sample(X_train, y_train, sample_mode='down')

print(y_train_dwn.value_counts())
print(y_train_dwn.value_counts(normalize=True))


# In[15]:


rfc_model = RandomForestClassifier(n_estimators = 200, max_depth = 15, random_state=123)

rfc_model.fit(X_train_dwn, y_train_dwn)
eval_model(rfc_model, X_test, y_test)


# In[16]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Data Preprocessing
# Separate the features and the target variable
X = imr_df.drop(['Child_Alive_or_Dead', 'Age_of_Child_at_Death', 'Age_at_death(Days)'], axis=1)

# Introduce Noise to Labels
noisy_labels = y.copy()
np.random.seed(123)  # for reproducibility
noise_indices = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create the preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, noisy_labels, test_size=0.3, random_state=123)

# Create a preprocessing and training pipeline
model_3 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=0))])

# Train the model
model_3.fit(X_train, y_train)

# Evaluate the model
y_pred = model_3.predict(X_test)
probs = model_3.predict_proba(X_test)

# Adjust Probability Threshold
custom_threshold = 0.6
noisy_preds = (probs[:, 1] > custom_threshold).astype(int)

# Output the Classification Report
report = classification_report(y_test, noisy_preds)
print(report)

eval_model(model_3, X_test, y_test)


# In[17]:


# Additional Evaluation Metrics
conf_mat = confusion_matrix(y_test, noisy_preds)
accuracy = accuracy_score(y_test, noisy_preds)
recall = recall_score(y_test, noisy_preds)
precision = precision_score(y_test, noisy_preds)
f1 = f1_score(y_test, noisy_preds)
print("\n")
print("Confusion Matrix:\n", conf_mat)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, probs[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC: ", roc_auc)


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preparing the data for regression analysis
# Selecting relevant columns for regression analysis and dropping NA values
regression_df = imr_df.dropna(subset=['Child_Alive_or_Dead', 'Age_of_Respondent', 'Education_Level', 'Wealth_Index', 'Residential_Type', 'Religion_code'])

# One-hot encoding categorical variables
categorical_features = ['Education_Level', 'Wealth_Index', 'Residential_Type', 'Religion_code']

# Creating a column transformer with OneHotEncoder
column_transformer = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# Defining the target variable and the features
X = regression_df[['Age_of_Respondent'] + categorical_features]
y = regression_df['Child_Alive_or_Dead']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline with column transformer and linear regression
pipeline = Pipeline(steps=[('preprocessor', column_transformer),
                           ('model', LinearRegression())])

# Fitting the model
pipeline.fit(X_train, y_train)

# Predicting the Test set results
y_pred = pipeline.predict(X_test)

# Calculating the coefficients, mean squared error, and the R^2 score
coefficients = pipeline.named_steps['model'].coef_
mean_squared_error = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)

# Printing the results
print('Coefficients:', coefficients)
print('Mean squared error:', mean_squared_error)
print('R^2 score:', r2_score)

# Visualizing the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing the time series data
time_series_df = imr_df.dropna(subset=['Year_of_Child_Death'])
time_series_df = time_series_df.groupby('Year_of_Child_Death')['Child_Alive_or_Dead'].count().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year_of_Child_Death', y='Child_Alive_or_Dead', data=time_series_df, marker='o')
plt.title('Child Deaths Over Years')
plt.xlabel('Year of Child Death')
plt.ylabel('Number of Child Deaths')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[21]:


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

# Testing for stationarity
result = adfuller(time_series_df['Child_Alive_or_Dead'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# If the time series is not stationary, we will difference it
if result[1] > 0.05:
    print('Time series is not stationary. Differencing will be applied.')
    time_series_df['Differenced'] = time_series_df['Child_Alive_or_Dead'].diff().dropna()
else:
    print('Time series is stationary.')

# Plotting the autocorrelation to identify potential ARIMA parameters
autocorrelation_plot(time_series_df['Child_Alive_or_Dead'])
plt.show()

# Fitting an ARIMA model
# For simplicity, we will use an ARIMA(1,1,1) model, which is a common starting point
model = ARIMA(time_series_df['Child_Alive_or_Dead'], order=(1,1,1))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecasting the next 5 years
forecast = model_fit.forecast(steps=5)
print('Forecast for the next 5 years:', forecast)


# In[ ]:





# In[ ]:





# In[ ]:




