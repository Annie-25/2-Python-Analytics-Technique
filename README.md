# Python-Analytics-Technique

Python Repo

## AIM 

The aim of this work is to perform data description, data preprocessing, data cleaning, data visualization, correlation analysis, unsupervised clustering, supervised classification and regression tree, and artificial neural network analysis. The goal is to gain insights from the dataset, explore patterns, detect outliers, identify relationships between variables, and develop predictive models. 

## Data Description 

Data Shape  

<img width="156" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/ef3045ab-7914-443d-aa85-730d2f2823e7">

Number of rows and columns of the data frame 

Data Size 

<img width="151" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/39d19091-857c-4b1b-b3cd-a3a3b356264e">

Total number of elements in the data frame 

Data Info

<img width="123" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/94123731-fa90-446c-9f59-942c9d9c546f">

Summary of the dataframe

## Data Preprocessing

The process of data preprocessing involves iteratively transforming raw data into forms that are comprehensible and usable. Raw datasets are commonly plagued by various issues, such as incompleteness, inconsistencies, lack of behavioral patterns, trends, and errors. To tackle these challenges, it is crucial to perform preprocessing, which can effectively handle missing values and inconsistencies. Data cleansing, reduction, scaling, transformation, and partitioning are the five main processes that make up constructing operational data preprocessing.

Checking for null values

<img width="150" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/8dd81a6b-538c-46aa-9ecb-a2db27b3d9a9">

Checking total number of null values for each column

<img width="139" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/f9754ccb-4acd-4a70-87e1-c0ac181e87e2">

Drop Null Method

select the 'id' column

<img width="136" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/8d5c3d2b-7dde-4d7d-99a2-a21fd97c1489"> 

To drop the column 'id' from the data frame while maintaining other columns (for df.drop) 

<img width="317" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/6bd6147c-c063-4ac3-ab0c-d2d449ac3daa"> 
 
## Data Distribution 

To produce a descriptive statistic for the data frame

<img width="135" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/29e70edf-ec35-48ef-8d28-dd3fbe4f982d">
   	 
## Data visualization 
Data visualization has various applications, including but not limited to aiding in the cleaning of data, exploring its structure, identifying outliers and anomalies in groups, recognizing trends and clusters, detecting local patterns, evaluating the output of models, and presenting results effectively. 

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/00912adb-7fb5-464b-a021-cb7610897894)

B - Benign 

M - Malignant 

The entire code is to create a pie plot/diagram for the column, 'diagnosis' with respect to the frequency of the unique values in the data frame. 

Output 1

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/d3a4e540-4c15-44ef-9f1f-f4a42c80b0fe)

Finding 1 

The pie chart shows the distribution of patients with malignant and benign cancer. According to the results, 63% of the patients suffered from benign cancer while 37% suffered from malignant cancer. 

Code Snippet 2: Box Plot 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/718bd7a9-c5a9-4aee-ad9d-89260de47111">

To plot a boxplot with 2 columns, diagnosis(x) and radius_mean(y)using sns library

Output 2 

 ![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/db19bb7f-d5d0-4e43-83a2-aa426e8d86e4)

Code Snippet 3: Histogram Visualization  

1. Individual plotting
   
![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/66d1406b-4e06-4adf-9ab7-296009227ee0)

a.to plot histograms for every column in the data frame 

b.to visualize the graphs 
 
 
2. Group plotting
   
<img width="471" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/be2b13a3-f404-4636-b0e9-2dca574df217">

This function is also to plot a histogram for all the columns at the same time/simultaneously. 
 
Output 3

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/24161e04-00d0-4640-8c0c-8998c239a5bd">
  
<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/a656ce60-69f4-4d78-a2c7-95e3d3e93008">
  
Finding 3 

Area 

•	A greater number of the cells recorded an area_mean between 0-1000 while a few recorded between 1000 - 2000.

•	All cells recorded an area_se between 0 - 200. 

•	Majority of cells recorded an area_worst between 0 – 2000 with a minimum number from 2000 but less than 4000. 

Compactness 

•	Majority of cells recorded compactness_mean between 0 – 0.2 but less than 0.2 with a minority recording between 0.2 to slightly above 0.3

•	Majority of cells recorded compactness_se from 0.00 – 0.05 with a minority recording between 0.05 to slightly above 0.10

•	Majority of cells recorded compactness_worst between 0.0 – 0.5 with a minority recording between 0.5 – 1.0

Concave Points 

•	Majority of cells recorded concave points_mean between 0.0 – 0.1 but less than 0.2 while a few recorded between 0.1 – 0.2

•	A greater number of the cells recorded a concave points_se between 0.00 – 0.02 with a minimum number from 0.02 but less than 0.04

•	Majority of cells recorded concave points_worst between 0 – 0.2 while a minority recording from 0.2 but less than 0.3

Concavity

•	A greater number of the cells recorded concavity_mean between 0.0 – 0.2 with a minimum number from 0.2 to slightly above 0.4

•	All cells recorded concavity_se between 0.0 to 0.2

•	Majority of cells recorded concavity_worst from 0 – 0.5 with a minority recording above 1.0 Diagnosis.

•	Majority of cells were 0 indicating benign cells while the minority were 1 indicating malignant cells. 

Fractal Dimension 

•	Many cells had fractal_dimension_mean between 0.00 – 0.08 with a few above 0.08

•	Many of the cells had fractal_dimension_se between 0.00 – 0.01 with the least number just slightly above 0.02

•	Majority also had fractal_dimension_worst between 0.05 to 0.10 with the least recording 0.20 Perimeter.

•	Majority of cells recorded a perimeter_mean, between 0 to 150 while the least recorded above 150.

•	Almost all cells had perimeter_se from 0 – 10

•	Almost all of the cells recorded perimeter_worst between 0 – 200 with very few cells recording above 200. 

Radius 

•	Majority of cells have a radius_mean between 0 – 20 with a minority above 20 

•	Majority of cells have a radius_se between 0 – 1 with a minority above 1 but less than 2 

•	Majority of cells have a radius_worst between 0 – 30 with a minority above 30

Smoothness 

•	Majority of cells have a smoothness_mean between 0 – 0.12 with a minority above 0.15 • Majority of cells have a smoothness_se between from 0 to 0.01 with a minority between 0.01 - slightly above 0.02

•	Majority of cells have a smoothness_worst between 0 – 0.20 with a minority above 0.20 Symmetry

•	Most cells had a symmetry_mean between 0.1 – 0.2 while the minority between 0.2 - 0.3

•	Most cells had a symmetry_se between 0 – 0.3 while a few had between 0.4 – 0.6

•	Most cells had a symmetry_worst between 0 – 0.4 while a few had between 0.4 – 0.6

Texture 

•	A greater number of the cells recorded a texture_mean between 10 – 30 with minority between 30 and 40. 

•	A greater number of the cells recorded a texture_se between 0 – 2 with the minority 4 as texture_se. 

•	Almost all cells recorded a texture_worst between 0 - 40 and the rest being above 40. 
 
Curve Plot

<img width="511" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/9329a19e-7007-4160-8e0e-ba5bdecf685a">

This function is also to plot a curve for all the columns at the same time/simultaneously. 
 	 
Code Snippet 4: 2-Dimensional Histogram 
 
![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/198fed86-3a54-424c-a676-40a9e4eb47e9)

Output 4 

<img width="282" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/a75c6d23-dba6-4769-a6ce-5663e1c8d57f">
  
Finding 4 

•	This is a 2-column at a time analysis (that is, only 2 columns are considered at a time). 

•	The shades of green represent the bin. Meaning that very light green (almost white) belong to bin 2 and so on. 

•	The deeper the shade of green, the higher the texture_mean and radius_mean.  
 
 	 
Correlation 
In general, correlation refers to a way of gauging the connection between variables. When data is correlated, the size of one variable's values is linked to changes in the size of another variable's values, either positively (when they change in the same direction) or negatively (when they change in opposite directions). (Schober, Boer and Schwarte, 2018). Code Snippet 
  
# To plot a correlation matrix for the entire data frame. 
 
  
# To create a heatmap of the correlation matrix above with annotations using the sns library 3. 
  
 
Output 
[Due to the size of the correlation matrix and heat map, a screenshot could not been attained. They can, however, be located in the jupyter file, output 17 and output 18] 
Finding 
The correlation matrix shows the correlation among several variables that determine the classification of the stages. Each cell indicates the correlation between two specific variables. For example, the correlation between radius_mean and perimeter_mean is 0.99 which indicates that they are strongly positively correlated while the correlation between the smoothness_se and texture_mean is 0.006614, indicating a weak positive relationship. It was also noted that some variables, for instance, texture_se and area_mean are negatively correlated. 
Scatter_Matrix
  
# To import scatter matrix function from the pandas.plotting library 
# Attributes are the columns to be considered for the scatter matrix. 
# To plot a scatter matrix for the above attributes with a figure size (20x20) and color (blue) 
 
 	  
Supervised Tasks 
Classification and Regression Tree 
Supervised learning is a machine learning method that involves training a model using labeled data to predict unknown values. The training data comprises a set of input-output pairs that the model uses to learn and make accurate predictions on new data. (Chorbngam, Chawuthai and Anantpinijwatna, 2021).  
 	  
Regression 
Code Snippet 
  
# Library to train, test, and split the data frame for further analysis. 
# library to create a linear regression for the model 
# library to encode the unique values within a column/data frame. 
# Total number of null values within the data frame 
  
# Function to encode the unique values (Malignant & Benign) within the "diagnosis" column as 1 and 0 respectively. 
  
# To convert the string values to float with the column, "diagnosis" 
# To split the data into test and train datasets of x and y 
# To draw a linear regression 
# To fit the x and y variables 
# To predict y (target) variables (diagnosis) 
# To display the model coefficients. 
# To display the model intercept. 
  
# To print the formula y=mx+c (linear regression formula) with both x and y variables. 
  
# To predict y variable using a new data frame of x variables. 
  
# Metrics libraries to calculate mean squared error, r2 score and mean absolute error of the considered data frame. 
  
  
Comment 
To find the regression of the model, we first found the and the intercept and the coefficient to complete the equation; y=mx+c. We also run a code to check if the model fits the data frame. After which we predicted the model with new data value which gave us;  
 indicating a malignant or cancerous cell. 
Classifiers 
Classification involves determining the category or subpopulation to which a new observation belongs, using a training dataset that consists of existing observations or instances - (Lamba, Hsu and Alsadhan, 2021). 
Code Snippet 
  
Decision Tree Classifier 
Code Snippet 
  
  
  
Comment 
The breast cancer dataset was visualized using a decision classifier that was trained with a maximum depth of 8. In the visualization, each square represents a node in the decision tree, and at the bottom of each square, there are two leaf nodes. The square contains information such as the gini impurity, the total number of samples per class, and the dominant class for the subset of training data at that particular node. This dominant class also represents the prediction output of the node. 
Random Forest Classifier 
Code Snippet 
  
# Library to call the random forest classifier. 
# Function to call the random forest classifier model 
# Fitting x and y variables to the random forest classifier. 
  
# To predict y (target) variable using random forest model. 
 
  
Output
  
Comment 
The color of the squares in the visualization corresponds to the level of class purity. The benign class is represented by the color blue, while the malignant class is represented by the color orange. The darkness of the color indicates the purity of the class, with darker squares indicating a higher level of class purity. 	 
Performance Metric Using Accuracy Score 
Code Snippet 
  
# Libraries to conduct performance metric to select best fit model using accuracy score. 
  
# To display, the classifier name and model 
# To fit trained x and y to each model 
# To print accuracy score for each model using x and y test values. 
Output
  
Comment 
In the given results: 
 
-	LogisticRegression achieved a test accuracy score of 0.9296703296703297, which means it correctly predicted the labels for approximately 92.97% of the test samples. 
 
-	KNeighborsClassifier achieved a test accuracy score of 0.9230769230769231, indicating that it correctly predicted the labels for about 92.31% of the test samples. 
 
-	DecisionTreeClassifier achieved a test accuracy score of 0.8923076923076924, meaning it correctly predicted the labels for around 89.23% of the test samples. 
 
-	RandomForestClassifier achieved a test accuracy score of 0.9252747252747253, indicating that it correctly predicted the labels for approximately 92.53% of the test samples. 
 
Based on the accuracy scores, LogisticRegression achieved the highest accuracy score of 0.9296703296703297, indicating that it performed slightly better than the other models in terms of accuracy. Therefore, based on this evaluation metric, LogisticRegression can be considered the best fit among the given models. 
 	 
Unsupervised Tasks 
Clustering 
According to El Bouchefry and de Souza (2020) unsupervised learning is a type of machine learning technique that involves the exploration of patterns and relationships within unlabelled and uncategorized training data in order to uncover interesting and previously unknown insights.  
 Alashwal et al., (2019) explained that, clustering is an unsupervised method of machine learning that involves segregating a group of data points based on their degree of similarity, often measured through distance. The main objective of clustering is to identify subgroups within a diverse dataset in such a way that each cluster has more uniformity than the dataset as a whole. 
Code Snippet 
K-Means Clustering 
  
# To import KMeans clustering library. 
Finding 
The plot above indicates two columns, ‘diagnosis’ (x-axis) versus radius_mean (y-axis). In the plot are two classes from diagnosis column; 1 (Malignant) and 0 (Benign). Each color represents a cluster; red = cluster 1, green = cluster 2 and blue = cluster 3. 
 
Andrews Curves 
  
# To plot andrews curve using diagnosis column. 
Output
 	 
Finding 
From the plot above, each color denotes a class. The dull greenish color represents 1.0 (Malignant) while the brighter greenish color represents 0.0 (Benign). As such the lines within the curves which denotes samples from the same class have similar curves. 
 
  
# Index to select specific rows and columns/values from a data frame. 
 
Output 
 
  
# number of clusters and iterations to be performed 
# fitting the model to the new data frame 
  
# Defining the centroids 
 
 
Elbow Method 
 
# Considering the first and tenth columns for the clustering 
Output 
 
Finding 
From the plot, it is evident that the elbow drops at 3. This justifies that the number of clusters defined previously as ‘3’ is correct. 
Artificial Neural Network 
Basic Neural Network Code Snippet 
  
  
Comment 
Error 
MLPRegressor 
Code Snippet 
  
# Libraries needed to carry out a MLPRegressor network. 
    
  
  
  
  
Comment 
Since r2 is 0.56 denoting that it’s closer to 1, it indicates a better relationship between the predicted and actual values with a mean absolute error of 0.26 and a root mean squared error of 0.32. 






