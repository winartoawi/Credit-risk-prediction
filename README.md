# Machine-learning-Credit-risk-prediction

In the financial firms, credit or loan providers will apply a credit scoring assessment to evaluate the credit risk for their clients, ensuring the debt to be repaid. 
The credit risk assessment is an essential tool helping lenders in making a decision ensuring profitability in granting credit/loan. 

The proposed system consists of three types of predictive models below, to identify which clients are more likely to experience a financial crisis in the next two years. 
- k-Nearest Neighbour  
- Random Forest
- XGBoost

The development of the proposed model will aid financial firm’s decision making for credit granting with the help of risk management – credit risk. 
The dataset used in this project will be based on a kaggle dataset credit-risk from this link https://www.kaggle.com/c/GiveMeSomeCredit/overview

**Input** - Numerical variables (client`s background information e.g. no of times late, no of credit cards, etc)

**Output** - Binary classification (Defaulters or Non-defaulters - 0 or 1)

The proposed model will demonstrate the implementation of a pre-processing method to deal with missing values and unbalanced target classes due to the nature of confidentiality, further, integration of fine-tuning such as feature selection will be explored to produce the best combination of predictive performance. 


# Model background
- **k–Nearest Neighbor**

K-Nearest Neighbor is a supervised machine learning algorithm that uses for regression or classification purposes whereby the model predicts the classification of a new data based on the nearest neighbor data points using similarity measures (distance function). 

In the past research, many models of kNN are used for financial credit scoring assessments due to its statistical pattern recognition Pandey et al. (2017). 

In this work kNN will be built with the implementation of 5-fold cross validations. 

- **Random Forest**

Random forest is a supervised machine learning algorithm that took the concept of bagging and random subspace feature selection to merge individual decision tree Pandey et al. (2017). 

The use of random forest for credit risk model helps to significantly reduced the time spent on data management and data pre-processing steps, especially on a large-scale dataset.

- **XGBoost**

XGBoost is a supervised machine learning algorithm that is an enhancement of a tree-based algorithm. 

XGBoost model is developed based on a gradient boosting framework of which the models can reduce variance and bias due to the bagging and boosting concepts. 

Extreme Gradient Boosting is an advanced implementation of gradient boosting and can be used for both classification and regression purposes. 


# Data exploratory 
There are 2 continuous attribute values and discrete attribute for the rest, whereas attribute class mostly are nominal classes and interval.

![image](https://user-images.githubusercontent.com/43923087/129149543-d9131fc0-8d02-432b-ad2b-f351d24a68a1.png)

# Data structure
The 11 variables data type are listed down where majority if the datatype consists of integer, number and factor for target variable.

![image](https://user-images.githubusercontent.com/43923087/129151444-348c30c4-697e-4a3d-95c6-b14bf2c13108.png)


# Target variable data imbalances
The ratio between both classes shows an extreme gap by 93:6 ratio of 139 974 to 10 026 observations respectively.

![image](https://user-images.githubusercontent.com/43923087/129151631-f79ce2e4-c22d-47a1-85ca-80bb3aba3c9d.png)

# Collinearity analysis
high collinearity between number of times of late payment in 60 – 89 days and number of times of late payment in 90 days, therefore 1 of the variables will be removed and proceed to model developmen

![image](https://user-images.githubusercontent.com/43923087/129153653-c94f9608-c4fe-40f5-afae-517d9e695888.png)


# Results
Training Result

![image](https://user-images.githubusercontent.com/43923087/129167511-4c747a39-1faf-4a98-99da-b04e3526aef6.png)

Test Result

![image](https://user-images.githubusercontent.com/43923087/129167523-0e710863-263d-441e-8626-10b5c6cb34b4.png)

ROC Curve 
The ROC curve of which by observations the XGBoost ROC has a wider curve closed to the true positive rate among others with AUC of 0.847.

![image](https://user-images.githubusercontent.com/43923087/129167280-eb779d7e-bd0d-4487-995a-6045e28c3f3c.png)


# Reference
Pandey, T. N., Jagadev, A. K., Mohapatra, S. K., & Dehuri, S. (2017). Credit risk analysis using machine learning classifiers. 2017 International Conference on Energy, Communication, Data Analytics and Soft Computing (ICECDS) 
