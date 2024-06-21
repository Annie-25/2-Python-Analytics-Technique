# Python-Analytics-Technique

Python Repo

## Table of Contents

- [About the project](#about-the-project)
- [AIM](#aim)
- [Techniques Applied](#techniques-applied)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Data Distribution ](#data-distribution)
- [Data Visualization](#data-visualization)
- [Correlation](#correlation)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Artificial Neural Network](#artificial-neural-network)
  
## AIM 

The aim of this work is to perform data description, data preprocessing, data cleaning, data visualization,correlation analysis, unsupervised clustering, supervised classification and regression tree, and artificial neural network analysis. The goal is to gain insights from the dataset, explore patterns, detect outliers, identify relationships between variables, and develop predictive models. 

## Techniques Applied

correlation analysis, unsupervised clustering, supervised classification and regression tree, and artificial neural network analysis

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

     
## Data Visualization 
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
 
 	 
## Correlation

In general, correlation refers to a way of gauging the connection between variables. When data is correlated, the size of one variable's values is linked to changes in the size of another variable's values, either positively (when they change in the same direction) or negatively (when they change in opposite directions). 

Code Snippet 

Plot a correlation matrix for the entire data frame.

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/949b506e-33c4-4d07-b414-7e4176835f4f)

To create a heatmap of the correlation matrix above with annotations using the sns library 3. 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/6c59fe03-5b51-4c63-b457-60ecdcc12661">
   
Output 

[Due to the size of the correlation matrix and heat map, a screenshot could not been attained. They can, however, be located in the jupyter file, output 17 and output 18] 

Finding 

The correlation matrix shows the correlation among several variables that determine the classification of the stages. Each cell indicates the correlation between two specific variables. For example, the correlation between radius_mean and perimeter_mean is 0.99 which indicates that they are strongly positively correlated while the correlation between the smoothness_se and texture_mean is 0.006614, indicating a weak positive relationship. It was also noted that some variables, for instance, texture_se and area_mean are negatively correlated. 

SCATTER_MATRIX

-Import scatter matrix function from the pandas.plotting library.

-Attributes are the columns to be considered for the scatter matrix.

-Plot a scatter matrix for the above attributes with a figure size (20x20) and color (blue) 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/fc373f1c-ab6e-4f29-9360-37bea13e2c9c">

 	  
## Supervised Learning

CLASSIFICATION and REGRESSION 

Supervised learning is a machine learning method that involves training a model using labeled data to predict unknown values. The training data comprises a set of input-output pairs that the model uses to learn and make accurate predictions on new data.   
 	  
#Regression

Code Snippet 

<img width="319" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/f0dc6de4-5fa9-4b8a-a7e1-26ad7eec4fe8">

 The code is explained below:
 
-Library to train, test, and split the data frame for further analysis.

-Library to create a linear regression for the model 

-Library to encode the unique values within a column/data frame. 

-Total number of null values within the data frame 

<img width="369" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/3b8deb5d-6097-4f9e-8ee3-689633cb7fe9">

Encoding the unique values (Malignant & Benign) within the "diagnosis" column as 1 and 0 respectively.

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/e2f98e99-ac5f-4d6f-82b0-915e2b14ff4e">

The code is explained below:

1. To convert the string values to float with the column, "diagnosis"

2. To split the data into test and train datasets of x and y 

3. To draw a linear regression 

4. To fit the x and y variables 

5. To predict y (target) variables (diagnosis)

6. To display the model coefficients.

7. To display the model intercept. 

Print the formula y=mx+c (linear regression formula) with both x and y variables. 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/5bbe0168-6612-425c-af46-63036dcf67c0">

To predict y variable using a new data frame of x variables. 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/98c434d1-2b42-4b65-b921-af8e7df30bf5">

Metrics libraries to calculate mean squared error, r2 score and mean absolute error of the considered data frame. 
  
<img width="447" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/ed9e5e47-14f9-48d2-8c1e-3ecc00084009">

Number of rows and columns in the data
<img width="78" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/96020180-e23e-466b-8745-df9ab8a0dc44">

Print the mean squared error, r2 score and mean absolute error 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/65199301-a626-44c9-a4cf-af54d2f9eb49">


Comment 

To find the regression of the model, we first found the and the intercept and the coefficient to complete the equation; y=mx+c. We also run a code to check if the model fits the data frame. After which we predicted the model with new data value which gave us;

<img width="123" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/9bad121a-4935-4c10-a93c-a330bb7aab29">
indicating a malignant or cancerous cell. 


#Classifiers 

Classification involves determining the category or subpopulation to which a new observation belongs, using a training dataset that consists of existing observations or instances.

Code Snippet 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/001ebf17-f93a-4d02-b0ff-9b257e8e5354">

DECISION TREE CLASSIFIER 

Code Snippet 

Import the decision tree classifier model and fit it to the data
 
<img width="262" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/a0abf388-01c3-4a58-b2d1-8ede65310aa9">

Defining the names of the features on the tree

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/770d4625-3c6b-4c73-a4b6-d6f09031c6b4">

Predict with new values

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/c07a9ab5-36ab-4dfb-8db5-47baaf5df9c3">
  
Comment 

The breast cancer dataset was visualized using a decision classifier that was trained with a maximum depth of 8. In the visualization, each square represents a node in the decision tree, and at the bottom of each square, there are two leaf nodes. The square contains information such as the gini impurity, the total number of samples per class, and the dominant class for the subset of training data at that particular node. This dominant class also represents the prediction output of the node. 

RANDOM FOREST CLASSIFIER

Code Snippet 

<img width="310" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/3dd45231-cca7-490c-a756-1cfb4e810898">
 
-Library to call the random forest classifier. 

-Function to call the random forest classifier model 

-Fitting x and y variables to the random forest classifier. 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/963ecd8b-1088-4ebe-8d2d-ee3aec041125">

To predict y (target) variable using random forest model. 
 
Plotting the tree

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/90e4d1bc-11d8-4618-a5cc-65af3f7952d8">

Output

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/6c9fc467-efec-46ab-800e-06b1865a8ffa">

Comment 

The color of the squares in the visualization corresponds to the level of class purity. The benign class is represented by the color blue, while the malignant class is represented by the color orange. The darkness of the color indicates the purity of the class, with darker squares indicating a higher level of class purity. 

Performance Metric Using Accuracy Score

Code Snippet 

Import libraries to conduct performance metric to select best fit model using accuracy score.

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/7c75dc71-8695-49a2-8832-203ee749df2d">

Finding out the classifier with highest accuracy score

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/36ec7acd-0941-43cd-b6ea-90e72bf412c6">

Output

<img width="247" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/065194ec-a682-467d-96b6-56c7190f1623">

Comment 

In the given results: 
 
-	LogisticRegression achieved a test accuracy score of 0.9296703296703297, which means it correctly predicted the labels for approximately 92.97% of the test samples. 
 
-	KNeighborsClassifier achieved a test accuracy score of 0.9230769230769231, indicating that it correctly predicted the labels for about 92.31% of the test samples. 
 
-	DecisionTreeClassifier achieved a test accuracy score of 0.8923076923076924, meaning it correctly predicted the labels for around 89.23% of the test samples. 
 
-	RandomForestClassifier achieved a test accuracy score of 0.9252747252747253, indicating that it correctly predicted the labels for approximately 92.53% of the test samples. 
 
Based on the accuracy scores, LogisticRegression achieved the highest accuracy score of 0.9296703296703297, indicating that it performed slightly better than the other models in terms of accuracy. Therefore, based on this evaluation metric, LogisticRegression can be considered the best fit among the given models. 
 	 
## Unsupervised Learning 

Unsupervised learning is a type of machine learning technique that involves the exploration of patterns and relationships within unlabelled and uncategorized training data in order to uncover interesting and previously unknown insights.  

#CLUSTERING

Clustering is an unsupervised method of machine learning that involves segregating a group of data points based on their degree of similarity, often measured through distance. The main objective of clustering is to identify subgroups within a diverse dataset in such a way that each cluster has more uniformity than the dataset as a whole. 

Code Snippet

K-Means Clustering 

Importing K-Means library

<img width="211" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/3c1b45ae-ab3e-4fc1-b4bd-7fb447df1fd9">

Findings

The plot above indicates two columns, ‘diagnosis’ (x-axis) versus radius_mean (y-axis). In the plot are two classes from diagnosis column; 1 (Malignant) and 0 (Benign). Each color represents a cluster; red = cluster 1, green = cluster 2 and blue = cluster 3. 
 
Andrews Curves 

<img width="266" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/c702699d-0499-4415-bdbc-a1c80b7d4463">
 
Output

<img width="312" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/fd9de93e-4358-4a81-83d9-c52bae7f1e31">

Findings

From the plot above, each color denotes a class. The dull greenish color represents 1.0 (Malignant) while the brighter greenish color represents 0.0 (Benign). As such the lines within the curves which denotes samples from the same class have similar curves. 
 
Creating A New Dataframe

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/b31c975b-d5ee-43bc-afe1-fc893fcd5f52)
 
Output 
 
![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/fd711246-12de-410f-a57d-765f983f1bd8)

-number of clusters and iterations to be performed 
-fitting the model to the new data frame 

<img width="224" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/4e0ac44b-093e-40e1-8315-88bc641db295">
  
Defining the centroids 
 
<img width="195" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/aebb314a-8837-40f9-8d21-a679eee7ba53">
 
Plotting the clustered results

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/93b001f7-0aeb-4f6a-ba6e-2646c6e4a5ca)

Output

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/e18e0966-ae4c-48de-b247-e1b0d5144376)

ELBOW METHOD

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/17a3ce8a-bde7-494b-ae5c-b34e81a978df)

Considering the first and tenth columns for the clustering 

Output 

![image](https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/5c192b69-db85-407b-b7f7-e1a05ecc145a)

Findings

From the plot, it is evident that the elbow drops at 3. This justifies that the number of clusters defined previously as ‘3’ is correct. 

## Artificial Neural Network

#BASIC NEURAL NETWORK

Code Snippet 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/23cb809c-89ba-445b-adfa-69a8e8c42b65">

 
<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/43cd08cf-380f-4b71-9192-de099c4e65f8">
 
Comment

Error 

#MLPREGRESSOR

Code Snippet 

<img width="468" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/011f9d20-c00e-4006-95f9-b939d1c0eb37">

Libraries needed to carry out a MLPRegressor network. 
    
<img width="134" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/fa4e1b9b-8942-41c4-b8b2-5bd05b57feab">

summary of x-train
  
<img width="69" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/c32c850d-2ee1-46a7-90e4-17140e103c5b">

 viewing y-train
 
<img width="137" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/de066adf-2fb3-49dc-a0dc-3995d8cc1780">

  summary y-train
  
<img width="306" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/1dd98319-1f20-4f68-95ca-f827ba852a81">

Creating the MLPRegressor model

<img width="192" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/d25e621c-09b3-4c15-9073-484b197287fa">

Fitting and predicting the model

<img width="292" alt="image" src="https://github.com/Annie-25/2-Python-Analytics-Technique/assets/173366226/88e5bb50-7181-43f6-9722-41df35c85205">

Printing the mean squared error (MAE), root mean square error (RMSE) and r-square

Comment

Since r2 is 0.56 denoting that it’s closer to 1, it indicates a better relationship between the predicted and actual values with a mean absolute error of 0.26 and a root mean squared error of 0.32. 






