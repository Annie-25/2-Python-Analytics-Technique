#!/usr/bin/env python
# coding: utf-8

# In[1]:


# These are libraries needed to read, explore/analyze and visualize the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/TOSHIBA/Downloads/data.csv") # reading the csv file/dataframe
df.head() # To show the first 5 rows of the dataframe


# # Data Description 

# In[3]:


df.shape # number of rows and columns of the dataframe


# In[4]:


df.size # total number of elements in the dataframe


# In[5]:


df.info() # prints detailed information about the dataframe


# # Data Cleaning

# In[6]:


df.isnull() # To check for null values


# In[7]:


df.isnull().sum() # To check total number of null values for each column


# ## Drop Null Method

# In[8]:


print(df['id']) # To print a column from the dataframe (in this case; 'id')


# In[9]:


df.drop(['id'],axis=1,inplace=True) # To drop the column 'id' from the data frame while maintaining other columns.


# In[10]:


df.head()


# # Data Distribution

# In[11]:


df.describe().T # To produce a descriptive statistics for the dataframe.


# # Data Visualization

# ## Diagnosis: Malignant vs Benign

# In[12]:


x=df["diagnosis"].value_counts().head(10) # value count() is used to check the frequency of each unique value within the dataframe.
print(x)
plt.figure(figsize=(8,8)) # to determine the size of the diagram/plot
cols_values = ['red','cyan'] # colors to be used to represent each group
plt.pie(x,labels=df["diagnosis"].value_counts().head(10).index,autopct="%0.0f%%",shadow=True,startangle=90,colors=cols_values);
plt.show()

# B - Benign
# M - Malignant

# The entire code is to create a pie plot/diagram for the column,'diagnosis' with respect to the frequency of the unique values in the 'diagnosis' dataframe.


# ## Box Plot

# In[13]:


sns.boxplot(data=df,x='diagnosis',y='radius_mean') # To plot a boxplot with 2 columns, diagnosis(x) and radius_mean(y)using sns library 


# ## Histogram Visualization

# In[14]:


df.hist(figsize = (20, 15)) # to plot histograms for every column in the dataframe
plt.show() # to visualize the graphs


# In[15]:


for i in ('radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean',
          'concave points_mean','symmetry_mean','fractal_dimension_mean'): 
    plt.figure()
    df[i][df['diagnosis']=='M'].plot.hist(alpha=0.5,color='red',title=i)
    df[i][df['diagnosis']=='B'].plot.hist(alpha=0.5,color='green')
# This function is also to plot a histogram for all the columns at the same time/simultaneously.
    plt.show()


# In[16]:


for i in ('radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean',
          'concave points_mean','symmetry_mean','fractal_dimension_mean'):
    df[i].plot.kde(title=i)
# This function is also to plot a curve for all the columns at the same time/simultaneously.    
    plt.show()


# ## 2-Dimensional Histogram

# In[17]:


x=df['texture_mean']
y=df['radius_mean']
plt.hist2d(x,y,bins=30,cmap='Greens') # To plot a 2d histogram for x and y values above
cb=plt.colorbar() # to add a color bar to serve as a key/legend
cb.set_label('counts in bin') # to label the color bar.


# # Correlation

# In[18]:


corr_matrix=df.corr() 
corr_matrix
# To plot a correlation matrix for the entire dataframe.


# In[19]:


plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True) # To create a heatmap of the correlation matrix above with annotations using the sns library.


# In[20]:


corr_matrix['smoothness_mean'].sort_values(ascending=False) 
# To correlate 'smoothness_mean' with the other columns without a specific order.


# ## Scatter Matrix

# In[21]:


# To import scatter matrix function from the pandas.plotting library
from pandas.plotting import scatter_matrix


# In[22]:


attributes=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean',
          'concave points_mean','symmetry_mean','fractal_dimension_mean']
# Attributes are the columns to be considered for the scatter matrix.

scatter_matrix(df[attributes],figsize=(20,20),color='#A029FA',hist_kwds={'color':['blue']})
# To plot a scatter matrix for the above attributes with a figure size (20x20) and color (blue)


# # Supervised Tasks

# ## Regression

# In[23]:


from sklearn.model_selection import train_test_split # Library to train, test, and split the data frame for further analysis.
from sklearn import linear_model # library to create a linear regression for the model
from sklearn.preprocessing import LabelEncoder # library to encode the unique values within a column/dataframe.


# In[24]:


df.isnull().sum().sum() # Total number of null values within the dataframe


# In[25]:


label_enncode = LabelEncoder()

df['diagnosis']= label_enncode.fit_transform(df['diagnosis'])
  
df['diagnosis'].unique()

# Function to encode the unique values (Malignant & Benign) within the "diagnosis" column as 1 and 0 respectively.


# In[26]:


df["diagnosis"] = [float(str(i).replace(",", "")) for i in df["diagnosis"]] 
# To convert the string values to float with the column, "diagnosis"


# In[27]:


x=df[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst','fractal_dimension_worst']]
y=df['diagnosis']


# In[28]:


x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2, random_state=42) # To split the data into test and train datasets of x and y.
model=linear_model.LinearRegression() # To draw a linear regression
model.fit(x,y) # To fit the xand y variables
y_hat=model.predict(x_test) # To predict y (target) variables (diagnosis)
print('Coefficients:',model.coef_) # To display the model coefficients.
print('Intercept:',model.intercept_) # To display the model intercept.


# In[29]:


print('diagnosis=%.4f +%.4f radius_mean+%.4f texture_mean+%.4f perimeter_mean+%.4f area_mean+%.4f smoothness_mean+%.4f compactness_mean+%.4f concavity_mean+%.4f concave points_mean+%.4f symmetry_mean+%.4f fractal_dimension_mean+%.4f radius_worst+%.4f texture_worst+%.4f perimeter_worst+%.4f area_worst+%.4f smoothness_worst+%.4f compactness_worst+%.4f concavity_worst+%.4f concave points_worst+%.4f symmetry_worst+%.4f fractal_dimension_worst')
# To print the formula y=mx+c (linear regression formula) with both x and y variables. 


# In[30]:


model.predict([[18.09, 11.58, 132.5, 1005, 0.1204, 0.2566, 0.2997, 0.1391, 0.3029, 0.05971, 0.680, 0.7023, 7.679, 173.3, 0.005389, 0.02814, 0.03273, 0.00686, 0.01003, 0.015183, 26.28, 10.83, 144.8, 2000, 0.1262, 0.4657, 0.5119, 0.1624, 0.4551, 0.1200]])
# To predict y variable using a new dataframe of x variables.


# In[31]:


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from math import sqrt
# Metrics libraries to calculate mean squared error, r2 score and mean absolute error of the considered dataframe.


# In[32]:


y_test.shape


# In[33]:


y_hat.shape


# In[34]:


print('Coefficient of determination(R^2):%.3f'%r2_score(y_test,y_hat))
print('Mean squared error(MSE):%.3f'%mean_squared_error(y_test,y_hat))
print('Root mean squared error(RMSE):%.3f'%sqrt(mean_squared_error(y_test,y_hat)))


# # Classifiers

# In[35]:


x=df[['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean', 'concavity_mean', 'concave points_mean', 
      'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
      'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
      'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst','fractal_dimension_worst']]
y=df['diagnosis']


# ## Decision Tree Classifier

# In[36]:


from sklearn import tree # Library to carry out a decision tree classifier.


# In[37]:


clf=tree.DecisionTreeClassifier() # Function to Decision Tree Classifier 


# In[38]:


clf=clf.fit(x,y) # Fitting the x and y variables to the classifier

tree.plot_tree(clf) # To plot the decision tree


# In[39]:


tree.plot_tree(clf,feature_names=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean', 
                                  'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 
                                  'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
                                  'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                                  'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst','fractal_dimension_worst'],class_names=['1','0'])


# In[40]:


clf.predict([[18.09, 11.58, 132.5, 1005, 0.1204, 0.2566, 0.2997, 0.1391, 0.3029, 0.05971, 0.680, 0.7023, 7.679, 173.3, 0.005389, 0.02814, 0.03273, 0.00686, 0.01003, 0.015183, 26.28, 10.83, 144.8, 2000, 0.1262, 0.4657, 0.5119, 0.1624, 0.4551, 0.1200]])
#To predict y (target) variable using the decision tree classifier.


# ## Random Forest Classifier

# In[41]:


from sklearn.ensemble import RandomForestClassifier # Library to call the random forest classifier.


# In[42]:


rf = RandomForestClassifier() # Function to call the random forest classifier model
rf.fit(x,y) #fitting x and y variables to the random forest classifier.


# In[43]:


rf.predict([[18.09, 11.58, 132.5, 1005, 0.1204, 0.2566, 0.2997, 0.1391, 0.3029, 0.05971, 0.680, 0.7023, 7.679, 173.3, 0.005389, 0.02814, 0.03273, 0.00686, 0.01003, 0.015183, 26.28, 10.83, 144.8, 2000, 0.1262, 0.4657, 0.5119, 0.1624, 0.4551, 0.1200]])
# To predict y (target) variable using random forest model.


# In[44]:


from sklearn.tree import plot_tree # Library to plot random forest 

fig=plt.figure(figsize=(15,10))
for i in range(3):
    plot_tree(rf.estimators_[i],
         feature_names=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean', 
                                  'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 
                                  'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
                                  'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                                  'compactness_worst', 'concavity_worst','concave points_worst', 'symmetry_worst','fractal_dimension_worst'],
          class_names=['1','0'],
         filled=True,impurity=True,
         rounded=True)


# ## Performance Metrics Using Accuracy Score

# In[45]:


# Libraries to conduct performance metric to select best fit model using accuracy score.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[46]:


key = ['LogisticRegression','KNeighborsClassifier','DecisionTreeClassifier','RandomForestClassifier']
value = [LogisticRegression(max_iter=2500), 
         KNeighborsClassifier(n_neighbors = 5, weights ='uniform'), 
         DecisionTreeClassifier(random_state=10), 
         RandomForestClassifier(n_estimators=50, random_state=0)]
classifiers = dict(zip(key,value))
classifiers


# In[47]:


test_accuracy = [] 

for classifier_name,classifier_model in classifiers.items() : # To display, the classifier name and model
    
    classifier_model.fit(x_train,y_train) # To fit trained x and y to each model
    
       
    print('\n',classifier_name) 
    
    print("Test Accuracy Score :",accuracy_score(y_test,classifier_model.predict(x_test))) 
    
    test_accuracy.append(accuracy_score(y_test,classifier_model.predict(x_test))*100)
    
    # To print accuracy score for each model using x and y test values.


# # Unsupervised Tasks

# ## Clustering

# ### K-Means Clustering

# In[48]:


from sklearn.cluster import KMeans 
# To import KMeans clustering library.


# ### Andrews_Curves

# In[94]:


pd.plotting.andrews_curves(df,'diagnosis') # To plot andrews curve using the 'diagnosis' column.


# In[50]:


x=df.iloc[:,0].values # iloc is an index to select specific rows and columns/values from a dataframe.


# In[51]:


x


# In[52]:


y=df.iloc[:,1].values


# In[53]:


y


# In[54]:


plt.figure(figsize=(6,6))


# In[55]:


plt.scatter(x,y)
plt.xlabel('Diagnosis')
plt.ylabel('Radius_mean')
plt.title('Radius wise Diagnosis');


# In[56]:


new_df=pd.DataFrame(df.iloc[:,[0,1]])
new_df.head(5)
# Creating a new dataframe


# In[57]:


km=KMeans(n_clusters=3,max_iter=100) # number of clusters and iterations to be performed
km.fit(new_df) # fitting the model to the new dataframe


# In[58]:


centroids=km.cluster_centers_
centroids
# Defining the centroids


# In[59]:


labels=km.predict(new_df)
colmap={1:'r',2:'g',3:'b'}
colors=list(map(lambda x:colmap[x+1],labels))


# In[60]:


plt.scatter(new_df['diagnosis'],new_df['radius_mean'],color=colors,alpha=0.5)
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroids)
    
plt.xlabel('Diagnosis')
plt.ylabel('Radius_mean')
plt.title('Clustered Results');
plt.show()


# ### Elbow Method

# In[61]:


sse=[]
k_list=list(range(1,10)) # Considering the first and tenth columns for the clustering


# In[62]:


for k in k_list:
    km=KMeans(n_clusters=k)
    km.fit(new_df)
    sse.append(km.inertia_)


# In[63]:


plt.figure(figsize=(6,6))
plt.plot(k_list,sse,'-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')


# In[64]:


class NeuralNetwork():
    
    def _init_(self):
        
        np.random.seed(1)
        self.synaptic_weights=2*np.random.random((3,1))-1
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivatives(self,x):
        return x*(1-x)
    
    def train(self,training_inputs,training_outputs,training_iterations):
        for iteration in range(training_iterations):
            output=self.think(training_outputs)
            error=training_outputs-output
            adjustments=np.dot(training_inputs.T,error*self.sigmoid_derivatives(output))
            self.synaptic_weights += adjustments
    def think(self,inputs):
        inputs=inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))
        return output
    
    if __name__=="__main__":
        neural_network= NeuralNetwork()
        
        print('Beginning Randomly Generated Weights:')
        print(neural_network.synaptic_weights)
        
        
        training_inputs=np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])
        
        training_outputs=np.array([[0,1,1,0]]).T
        
        neural_network.train(training_inputs,training_outputs,15000)
        
        print('Ending Weights After Training:')
        print(neural_network.synaptic_weights)
        
        user_input_one=str(input('User Input One: '))
        user_input_two=str(input('User Input Two: '))
        user_input_three=str(input('User Input Three: '))
        
        print('Considering New Situation:', user_input_one,user_input_two,user_input_three)
        print('New Output Data:')
        print(neural_network.think(np.array[user_input_one,user_input_two,user_input_three]))
        print('Finally done!')


# ## MLPRegressor

# In[65]:


# Libraries needed to carry out a MLPRegressor network.
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# In[84]:


x=df[['radius_mean','perimeter_mean','texture_mean']]
y=df['diagnosis']
from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2, random_state=42)


# In[85]:


# Training the dataset
x_train


# In[68]:


y_train


# In[86]:


# Viewing the statistics
x_train.describe()


# In[70]:


y_train.describe()


# In[79]:


model=MLPRegressor(hidden_layer_sizes=(64,64,64),# controls the architecture of the network.
                  activation='relu', # hidden layer activation function
                  random_state=42,# Allows for reproducible results, controls random number generation for weights and biases.
                   max_iter=2000)#controls max number of iterations that the model will go if convergeance is not met prior.


# In[80]:


model


# In[87]:


model.fit(x_train,y_train)
y_pred=model.predict(x_test) # predict on the validation data


# In[88]:


y_pred


# In[89]:


mae=metrics.mean_absolute_error(y_test,y_pred) # measure of the absolute differences between predicted and actual value.
mse=metrics.mean_squared_error(y_test,y_pred) 
rmse=mse**0.5 # indicates the magnitude of the predicted error.
r2=metrics.r2_score(y_test,y_pred) # The  closer the relationship is to 1, the better the relationship.

print(f"""
MAE:\t{mae:.2f}
RMSE:\t{rmse:.2f}
r2:\t{r2:.2f}
""")


# In[ ]:




