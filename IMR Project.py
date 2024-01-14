#!/usr/bin/env python
# coding: utf-8

# #  Comparative Analysis between Multiple Regression and Time Series Approaches to Factors Influencing Infant Mortality Rates in Kenya.
# 
# ## 1 Introduction
# 
# Infant mortality is a major public health problem in Kenya, with an estimated 44 deaths per 1,000 live births in 2022. This project aims to identify the factors that influence infant mortality rates in Kenya using two different statistical approaches: multiple regression and time series analysis.
# 
# Multiple regression is a statistical technique that can be used to model the relationship between a dependent variable and multiple independent variables. In this project, the dependent variable is infant mortality rate and the independent variables include factors such as maternal education, household wealth, birth order, and place of residence.
# 
# Time series analysis is a statistical technique that can be used to model the behavior of a variable over time. In this project, time series analysis will be used to model infant mortality rates in Kenya over time and to identify trends and patterns.
# 
# The findings of this project will be used to develop recommendations for public health interventions to reduce infant mortality rates in Kenya.

# ## 2 Loading the Data

# In[1]:


# Import necessary libraries/modules
import os 
import re  
import pandas as pd  
import numpy as np 
import pyreadstat
from tabulate import tabulate  

# Specify constant parameters
random_seed = 123  # Set a random seed for reproducibility.
working_directory = r"C:\\Users\\Vincent\\Desktop\\Project\\Final_Year_Project"  
os.chdir(working_directory) 
pd.set_option('display.max_columns', None)  # Set a pandas option to display all columns when printing DataFrames.

# Read data from a Stata (.dta) file using the pyreadstat library
df, meta = pyreadstat.read_dta("KEKR8AFL.DTA")  
df = pd.DataFrame(df) 
df.head()  


# ### 2.1 Glimpse at Data

# In[2]:


# View the shape of the DataFrame
data_shape = df.shape
print("Data Shape (Rows, Columns):", data_shape)


# The shape of our data is described as 19,530 rows and 1,290 columns, signifying a substantial dataset with a considerable number of observations (rows) and a wide range of features or variables (columns). This suggests that our data is extensive and likely contains a wealth of information for analysis and exploration

# In[3]:


# Check the structure of each column in the DataFrame. Display the data types of the first and last 10 variables
column_data_types = df.dtypes
print("Data Types of the First 10 Variables:")
print(column_data_types[:10])

print("\nData Types of the Last 10 Variables:")
print(column_data_types[-10:])


# ## 3 Data Wrangling and Cleaning
# 
# ### 3.1 Extracting the required variables

# In[4]:


# List of specific variable names to extract
variable_names = ["caseid", "v012", "v106", "v501", "v190a", "v025", "v102",  "v024", "v130", "b1", "b4",  "b5", "b2", "b6", "b7"]
# This list contains the names of the specific variables we have extracted from the original DataFrame.

# Creating a new DataFrame with only the selected variables
selected_variables_df = df[variable_names]

# Setting display options for the DataFrame
pd.set_option('display.max_columns', None)  

# Print the entire content of the selected_variables_df DataFrame
selected_variables_df.head()


# In[5]:


# Dictionary for variable renaming
variable_name_mapping = {
    "caseid": "CaseID",
    "v012": "Age_of_Respondent",
    "v106": "Education_Level",
    "v501": "Marital_Status",
    "v025": "Residential_Type",
    "v190a": "Wealth_Index",
    "v102": "Place_of_Residence",
    "v024": "County_code",
    "v130": "Religion_code",
    "b1": "Month_of_Birth",
    "b4": "Sex_of_child",
    "b5": "Child_Alive_or_Dead",
    "b2": "Year_of_Child_Birth",
    "b6": "Age_of_Child_at_Death",
    "b7": "Calculated_Age_at_Death(Months)",
}

# Rename the variables in the DataFrame
IMR_dataset = selected_variables_df.rename(columns=variable_name_mapping)

# Display the DataFrame in a more readable tabular format
IMR_dataset.head()


# ### 3.2 Renaming the Selected Columns
# 
# #### Below is a renaming dictionary
# 
# v012 - Age of the respondent Renaming: Age_of_Respondent
# 
# v106 - Highest level of education Completed Renaming: Education_Level
# 
# v501 - Marital status Renaming: Marital_Status
# 
# v025 - Residential type Renaming: Residential_Type
# 
# v190a - Wealth index Renaming: Wealth_Index
# 
# v102 - Place of residence Renaming: Place_of_Residence
# 
# v024 - County / Region code
# 
# v130 - Religion code
# 
# b1 - The month of birth of the child Renaming: Month_of_Birth
# 
# b4 - Sex of the child
# 
# b5 - Whether the child was alive or dead Renaming: Child_Alive_or_Dead
# 
# b2 - Month of child's death Renaming: Month_of_Child_Death
# 
# b6 - Age of child at death as reported by the respondent Renaming: Age_of_Child_at_Death_Reported
# 
# b7 - Calculated age at death Renaming: Calculated_Age_at_Death

# View the shape of our new Dataframe

# In[6]:


data_shape = IMR_dataset.shape
print("Data Shape (Rows, Columns):", data_shape)


# The dimensions of our data, comprising _19,530_ rows and _12_ columns, signify a refined dataset. With a more focused set of 12 variables, our data is now streamlined and ready for in-depth analysis.

# 
# 
# __Exporting the data in csv formart__

# In[7]:


#IMR_dataset.to_csv("IMR_dataset.csv", index=False)


# ## Tyding data

# In[8]:


#checking the number of missing values
print(IMR_dataset.isnull().sum())


# The dataset seems relatively clean, with no missing values in most columns. However, the columns "Age_of_Child_at_Death" and "Calculated_Age_at_Death(Months)" have a significant number of missing values, specifically 18,836. This is because these entries depends on wheather Child_Alive_or_Dead is 0. Imputing  these missing values is necessary for a more comprehensive analysis.

# In[9]:


# We will use -1 as our place holder for the null values
placeholder_value = -1  

IMR_dataset['Age_of_Child_at_Death'].fillna(placeholder_value, inplace=True)
IMR_dataset['Calculated_Age_at_Death(Months)'].fillna(placeholder_value, inplace=True)


# In[10]:


#checking for categorical data
print(IMR_dataset.dtypes)


# __We noticed that the column "Age_of_Child_at_Death" has coded values__. The first digit in each coded value indicates the unit of measurement (1 for days, 2 for months, 3 for years, and 9 for special responses). The last two digits represent the age at death in those units. We wrote a function that transforms the coded values into exact days and stores them in a column named "Age_at_death(Days)"
# 
# The function checks for imputed values (e.g., -1) and returns them as they are. For valid entries, it calculates the exact age at death in days based on the specified unit. If the unit is 1 (days), it returns the age as is. If it's 2 (months), it assumes 1 month is equivalent to 30 days. If it's 3 (years), it assumes 1 year is equivalent to 365 days. Special responses are marked as None, indicating that the exact number of days is unknown.

# In[11]:


# Creating a function to convert the coded values to exact days
def convert_to_days(row):
    age_of_child = str(row['Age_of_Child_at_Death'])
    
    # Check for imputed values (e.g., -1) and return them as they are
    if age_of_child.startswith('-'):
        return int(age_of_child)
    
    unit = int(age_of_child[0])
    age = int(age_of_child[1:])
    
    if unit == 1:
        return age
    elif unit == 2:
        return age * 30  # Assuming 1 month = 30 days
    elif unit == 3:
        return age * 365  # Assuming 1 year = 365 days
    else:
        return None  # Special response, exact number of days is unknown

# Add a new column 'Age_at_death' using the convert_to_days function
IMR_dataset['Age_at_death(Days)'] = IMR_dataset.apply(convert_to_days, axis=1)

# Display the updated DataFrame
IMR_dataset.head()


# In[12]:


IMR_dataset.shape


# We have observed that our dataset now has 19530 rows and 13 columns. We want to narrows down the dataset to focus on cases where the calculated age at death is within the first year of life, targeting only infants. 

# In[13]:


# Define a dictionary that maps accurate county codes to county names without leading '0'
county_code_to_name = {
    1: "Mombasa", 2: "Kwale", 3: "Kilifi", 4: "Tana River", 5: "Lamu", 6: "Taita Taveta", 7: "Garissa", 8: "Wajir", 9: "Mandera", 10: "Marsabit", 11: "Isiolo", 12: "Meru",
    13: "Tharaka-Nithi", 14: "Embu", 15: "Kitui", 16: "Machakos", 17: "Makueni", 18: "Nyandarua", 19: "Nyeri", 20: "Kirinyaga",
    21: "Murang’a", 22: "Kiambu", 23: "Turkana", 24: "West Pokot", 25: "Samburu", 26: "Trans-Nzoia", 27: "Uasin Gishu", 28: "Elgeyo-Marakwet",
    29: "Nandi", 30: "Baringo", 31: "Laikipia", 32: "Nakuru", 33: "Narok", 34: "Kajiado", 35: "Kericho", 36: "Bomet",
    37: "Kakamega", 38: "Vihiga", 39: "Bungoma", 40: "Busia", 41: "Siaya", 42: "Kisumu", 43: "Homa Bay", 44: "Migori",
    45: "Kisii", 46: "Nyamira", 47: "Nairobi County"
}

# Create a new variable "County" in the IMR_dataset and assign the corresponding county name for each county code
IMR_dataset["County"] = IMR_dataset["County_code"].apply(lambda code: county_code_to_name.get(code, "Unknown"))


# In[14]:


# Create a new column named 'religion'
#df['religion'] = None

# Recode religion to group Christian faiths
IMR_dataset['Religion_code'] = IMR_dataset['Religion_code'].replace({
    10: 1, 7: 2, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 9: 4, 96: 4
})


# In[15]:


IMR_dataset = IMR_dataset[IMR_dataset['Calculated_Age_at_Death(Months)'] < 12]
IMR_dataset.shape


# In[ ]:





# ## Data Visualization

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is named 'IMR_dataset'
sns.set(style="whitegrid")  # Set the style for the plot

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plotting a pie chart for 'Child_Alive_or_Dead'
IMR_dataset['Child_Alive_or_Dead'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=axes[0], shadow=True)
axes[0].set_title('Child Alive or Dead(1: Alive, 0: Dead)')
axes[0].set_ylabel('')

# Plotting a count plot for 'Child_Alive_or_Dead'
sns.countplot(x='Child_Alive_or_Dead', data=IMR_dataset, palette='viridis', ax=axes[1])
axes[1].set_title('Child Alive or Dead(1: Alive, 0: Dead)')

# Calculate the count of dead and alive children
counts = IMR_dataset['Child_Alive_or_Dead'].value_counts()

# Display the plot
plt.show()
print(f"Dead: {counts[0]} | Alive: {counts[1]}")


# __Clearly the data is extreemly skewed!!__ 
# 
# ### Approach
# 
# 1. We are going to perform feature engineering.
# 2. We will then compare what happens when using resampling and when not using it. We will test this approach using a simple logistic regression classifier.
# 3. We will evaluate the models by using some of the performance metrics .
# 4. We will repeat the best resampling/not resampling method, by tuning the parameters in the logistic regression classifier.
# 5. We will finally perform classifications model using other classification algorithms.

# In[17]:


sns.set(style="whitegrid")  # Set the style for the plot

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Mapping labels for better visualization
residential_type_labels = {1: 'Urban', 2: 'Rural'}
IMR_dataset['Residential_Type_Label'] = IMR_dataset['Residential_Type'].map(residential_type_labels)

# Plotting a bar chart for 'Residential_Type_Label' vs. 'Child_Alive_or_Dead'
IMR_dataset[['Residential_Type_Label', 'Child_Alive_or_Dead']].groupby(['Residential_Type_Label']).mean().plot.bar(ax=axes[0])
axes[0].set_title('Child Mortality by Residential Type')
axes[0].set_ylabel('Survival Probability')

# Plotting a count plot for 'Residential_Type_Label' vs. 'Child_Alive_or_Dead'
sns.countplot(x='Residential_Type_Label', hue='Child_Alive_or_Dead', data=IMR_dataset, palette='viridis', ax=axes[1])
axes[1].set_title('Child Mortality by Residential Type')
axes[1].set_xlabel('Residential Type')
axes[1].set_ylabel('Count')

# Display the plot
plt.show()


# In[18]:


# Set the style for the plot
sns.set(style="whitegrid")

# Creating subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Mapping labels for better visualization
religion_labels = {1: 'No Religion', 2: 'Muslim', 3: 'Christian', 4: 'Other'}
IMR_dataset['Religion_Label'] = IMR_dataset['Religion_code'].map(religion_labels)

# Calculate and plot survival probability as percentages
total_counts = IMR_dataset.groupby('Religion_Label')['Child_Alive_or_Dead'].count()
survival_probabilities = IMR_dataset.groupby('Religion_Label')['Child_Alive_or_Dead'].mean()
survival_percentage = survival_probabilities * 100

# Plot a bar chart for 'Religion_Label' vs. 'Survival Percentage'
survival_percentage.plot.bar(ax=axes[0], color='skyblue', edgecolor='k', alpha=0.7)
axes[0].set_title('Child Survival Probability by Religion')
axes[0].set_ylabel('Survival Probabilities')
axes[0].set_ylim(0, 100)

# Annotate the bars with survival percentages
for i, v in enumerate(survival_percentage):
    axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom')

# Plot a count plot for 'Religion_Label' vs. 'Child_Alive_or_Dead'
sns.countplot(x='Religion_Label', hue='Child_Alive_or_Dead', data=IMR_dataset, palette='viridis', ax=axes[1])
axes[1].set_title('Child Mortality by Religion')
axes[1].set_xlabel('Religion')
axes[1].set_ylabel('Count')

# Display the plot
plt.show()


# In[19]:


# Filter the dataset to include only rows where the child is dead
dead_children = IMR_dataset[IMR_dataset['Child_Alive_or_Dead'] == 0]

# Count the number of deaths in each county and sort them in descending order
county_death_counts = dead_children['County'].value_counts().sort_values(ascending=False)

# Create a colorful horizontal bar graph
plt.figure(figsize=(12, 8))
sns.barplot(y=county_death_counts.index, x=county_death_counts.values, palette='viridis')
plt.title('Infant Mortality by County')
plt.xlabel('Number of Deaths')
plt.ylabel('Counties')
plt.show()


# In[35]:


from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table_county = IMR_dataset.pivot_table(index='County', columns='Child_Alive_or_Dead', aggfunc='size', fill_value=0)

# Perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table_county)

# Determine significance
significance = 'Significant' if p < 0.05 else 'Not Significant'

# Output results
print('Chi-squared statistic:', chi2)
print('P-value:', p)
print('Degrees of freedom:', dof)
print('Expected frequencies:', expected)
print('Significance:', significance)


# In[ ]:





# In[20]:


selected_variables = ['Age_of_Respondent', 'Wealth_Index', 'Age_of_Child_at_Death', 'Calculated_Age_at_Death(Months)']

# Subset the DataFrame with selected variables
selected_data = IMR_dataset[selected_variables]

# Create a scatter plot matrix
scatter_plot_matrix = sns.pairplot(selected_data, diag_kind='kde', markers='o', palette='viridis')

# Adjust layout and display the plot
plt.subplots_adjust(top=0.95)
scatter_plot_matrix.fig.suptitle('Scatter Plot Matrix for Selected Variables', size=16)
plt.show()


# In[21]:


sns.set(style="whitegrid")  # Set the style for the plot

# Plotting a box plot for 'Age_of_Respondent' vs. 'Child_Alive_or_Dead'
plt.figure(figsize=(6, 3))
sns.boxplot(x='Child_Alive_or_Dead', y='Age_of_Respondent', data=IMR_dataset, palette='viridis')

# Adding labels and title
plt.xlabel('Child Alive or Dead (1: Alive, 0: Dead)')
plt.ylabel('Age of Respondent')
plt.title('Distribution of Age of Respondent by Child Alive or Dead')

# Display the plot
plt.show()


# 

# ## Modeling

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ####   Multiple Regression and Time Series

# ## Key Insights

# ## Conclusion

# ## References

# In[22]:


IMR_dataset['Year_of_Child_Death'] = IMR_dataset['Year_of_Child_Birth'] + np.ceil(IMR_dataset['Age_of_Child_at_Death'] / 12)

# Set Age_of_Child_at_Death to NaN where Child is Alive
IMR_dataset.loc[IMR_dataset['Child_Alive_or_Dead'] == 1, 'Age_of_Child_at_Death'] = np.nan

# Convert 'Year_of_Child_Death' to whole numbers, handling NaN values
IMR_dataset['Year_of_Child_Death'] = IMR_dataset['Year_of_Child_Death'].round().astype('Int64')

# Display the head of the modified dataframe
IMR_dataset.head()


# In[23]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
IMR_dataset.head()


# In[24]:


IMR_dataset.to_csv("IMR_dataset.csv", index=False)


# In[25]:


import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate

tqdm.pandas()

# Drop rows with null values in 'Year_of_Child_Death' as we can't use them for trend analysis
IMR_dataset = IMR_dataset.dropna(subset=['Year_of_Child_Death'])

# Aggregate the data by year
death_counts_by_year = IMR_dataset.groupby('Year_of_Child_Death')['Child_Alive_or_Dead'].progress_apply(lambda x: (x == 0).sum()).reset_index()

# Rename columns for clarity
death_counts_by_year.columns = ['Year', 'Number_of_Deaths']

# Display the aggregated data as a table
table = tabulate(death_counts_by_year, headers='keys', tablefmt='grid', showindex=False)

# Print the table
print(table)


# In[26]:


import pandas as pd
from IPython.display import display

# Mapping dictionaries for categorical variables
education_level_map = {0: 'No Education', 1: 'Primary', 2: 'Secondary', 3: 'Higher'}
marital_status_map = {0: 'Not Married', 1: 'Married'}
wealth_index_map = {1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'}
residential_type_map = {1: 'Urban', 2: 'Rural'}
child_alive_map = {1: 'Alive', 0: 'Dead'}

# Apply mapping to create new columns
IMR_dataset['Education_Level'] = IMR_dataset['Education_Level'].map(education_level_map)
IMR_dataset['Marital_Status'] = IMR_dataset['Marital_Status'].map(marital_status_map)
IMR_dataset['Wealth_Index'] = IMR_dataset['Wealth_Index'].map(wealth_index_map)
IMR_dataset['Residential_Type'] = IMR_dataset['Residential_Type'].map(residential_type_map)
IMR_dataset['Child_Alive_or_Dead'] = IMR_dataset['Child_Alive_or_Dead'].map(child_alive_map)

# Calculate the distribution of categorical variables
education_level_dist = IMR_dataset['Education_Level'].value_counts().reset_index().rename(columns={'index': 'Education Level', 'Education_Level': 'Count'})
marital_status_dist = IMR_dataset['Marital_Status'].value_counts().reset_index().rename(columns={'index': 'Marital Status', 'Marital_Status': 'Count'})
wealth_index_dist = IMR_dataset['Wealth_Index'].value_counts().reset_index().rename(columns={'index': 'Wealth Index', 'Wealth_Index': 'Count'})
residential_type_dist = IMR_dataset['Residential_Type'].value_counts().reset_index().rename(columns={'index': 'Residential Type', 'Residential_Type': 'Count'})
child_alive_dist = IMR_dataset['Child_Alive_or_Dead'].value_counts().reset_index().rename(columns={'index': 'Child Alive or Dead', 'Child_Alive_or_Dead': 'Count'})

# Display DataFrames
print('Education Level Distribution:')
display(education_level_dist)

print('\nMarital Status Distribution:')
display(marital_status_dist)

print('\nWealth Index Distribution:')
display(wealth_index_dist)

print('\nResidential Type Distribution:')
display(residential_type_dist)

print('\nChild Alive or Dead Distribution:')
display(child_alive_dist)


# In[ ]:





# In[27]:


from scipy.stats import chi2_contingency
# Calculate survival probability by sex
IMR_dataset['Sex_of_child'] = IMR_dataset['Sex_of_child'].map({1: 'Male', 2: 'Female'})
IMR_dataset['Child_Alive_or_Dead'] = IMR_dataset['Child_Alive_or_Dead']#.astype(int)
survival_by_sex = IMR_dataset.groupby('Sex_of_child')['Child_Alive_or_Dead'].mean().reset_index()
survival_by_sex['Survival_Probability_Percentage'] = survival_by_sex['Child_Alive_or_Dead'] * 100

# Perform a chi-squared test to determine if the difference in survival probability is significant
contingency_table = IMR_dataset.pivot_table(index='Sex_of_child', columns='Child_Alive_or_Dead', aggfunc='size', fill_value=0)
chi2, p, dof, expected = chi2_contingency(contingency_table)
significance = 'Significant' if p < 0.05 else 'Not Significant'

# Plotting the survival probability by sex with percentages
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex_of_child', y='Survival_Probability_Percentage', data=survival_by_sex)
plt.title('Survival Probability by Sex of Child (Percentage)')
plt.xlabel('Sex of Child')
plt.ylabel('Survival Probability (%)')
plt.ylim(0, 100)
for index, row in survival_by_sex.iterrows():
    plt.text(row.name, row.Survival_Probability_Percentage, f'{row.Survival_Probability_Percentage:.2f}%', color='black', ha="center")
plt.show()

# Output the survival probabilities, significance, and p-value
IMR_dataset_result = survival_by_sex.copy()
IMR_dataset_result['Significance'] = significance
IMR_dataset_result['p-value'] = p
IMR_dataset_result


# In[32]:


# Visualizing the Wealth Index
plt.figure(figsize=(10, 6))
IMR_dataset['Wealth_Index'].value_counts(sort=False).plot(kind='bar')
plt.title('Chances of Infant death Death by Wealth Index')
plt.xlabel('Wealth Index')
plt.ylabel('Number of Infant dead')
plt.xticks(rotation=0)
plt.show()


# In[ ]:




