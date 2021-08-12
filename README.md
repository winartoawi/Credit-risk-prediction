# winartoawi-Machine-learning-Credit-risk-prediction

In the financial domain/firms, credit or loan providers will apply a credit scoring assessment to evaluate the credit risk of the client, ensuring the debt to be repaid. 
The credit risk assessment is an essential tool helping lenders in making a decision ensuring profitability in granting credit/loan. 
The proposed system consists of three types of predictive models such as logistic regression (LR), Naïve Bayesian, and Artificial Neural Network (ANN) to identify which clients are more likely to experience a financial crisis in the next two years. 
The proposed model will demonstrate the implementation of a pre-processing method to deal with missing values and unbalanced target classes due to the nature of confidentiality, further, integration of fine-tuning such as feature selection will be explored to produce the best combination of predictive performance. The development of the proposed model will aid financial firm’s decision making for credit granting with the help of risk management – credit risk. 
The dataset used in this project will be based on a kaggle dataset credit-risk from this link https://www.kaggle.com/c/GiveMeSomeCredit/overview


# Model background
•	k–Nearest Neighbor (kNN)
K-Nearest Neighbor is a supervised machine learning algorithm that uses for regression or classification purposes whereby the model predicts the classification of a new data based on the nearest neighbor data points using similarity measures (distance function). In the past research, many models of kNN are used for financial credit scoring assessments due to its statistical pattern recognition Pandey et al. (2017). In this work kNN will be built with the implementation of 5-fold cross validations. 

•	Random Forest (RF)
Random forest is a supervised machine learning algorithm that took the concept of bagging and random subspace feature selection to merge individual decision tree Pandey et al. (2017). The use of random forest for credit risk model helps to significantly reduced the time spent on data management and data pre-processing steps, especially on a large-scale dataset.

•	XGBoost
XGBoost is a supervised machine learning algorithm that is an enhancement of a tree-based algorithm. XGBoost model is developed based on a gradient boosting framework of which the models can reduce variance and bias due to the bagging and boosting concepts. Extreme Gradient Boosting is an advanced implementation of gradient boosting and can be used for both classification and regression purposes. 


Data exploratory 

![image](https://user-images.githubusercontent.com/43923087/129149543-d9131fc0-8d02-432b-ad2b-f351d24a68a1.png)

Data structure

![image](https://user-images.githubusercontent.com/43923087/129151444-348c30c4-697e-4a3d-95c6-b14bf2c13108.png)

Histogram plot 

![image](https://user-images.githubusercontent.com/43923087/129152183-5d0f58f0-9ca3-440e-8d8f-177c9fbb2656.png)



Target variable data imbalances

![image](https://user-images.githubusercontent.com/43923087/129151631-f79ce2e4-c22d-47a1-85ca-80bb3aba3c9d.png)

Collinearity analysis

![image](https://user-images.githubusercontent.com/43923087/129153653-c94f9608-c4fe-40f5-afae-517d9e695888.png)



# Reference
Pandey, T. N., Jagadev, A. K., Mohapatra, S. K., & Dehuri, S. (2017). Credit risk analysis using machine learning classifiers. 2017 International Conference on Energy, Communication, Data Analytics and Soft Computing (ICECDS) 
