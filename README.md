**Credit Risk Analysis**<br/>
This project is focused on performing credit risk analysis using various machine learning algorithms to predict loan default status based on the given dataset. The code provided below demonstrates the steps involved in the analysis:</br>

**Dataset**
The dataset used for this analysis is stored in the file "credit_risk_dataset.csv". The dataset contains the following columns:<br/>

person_age: Age of the person<br/>
person_income: Income of the person<br/>
person_home_ownership: Type of home ownership (categorical)<br/>
person_emp_length: Employment length of the person<br/>
loan_intent: Purpose of the loan (categorical)<br/>
loan_grade: Grade of the loan (categorical)<br/>
loan_amnt: Loan amount<br/>
loan_int_rate: Interest rate of the loan<br/>
loan_status: Loan status (0 for non-default, 1 for default)<br/>
loan_percent_income: Percentage of income that the loan amount represents<br/>
cb_person_default_on_file: Credit bureau information (categorical)<br/>
cb_person_cred_hist_length: Credit history length of the person<br/>

**Data Preprocessing**<br/>
Checking for Null Values: The code starts by checking for null values in the dataset using df.isnull().sum(). Missing values in person_emp_length and loan_int_rate are found, and these rows are dropped using df = df.dropna(axis=0).<br/>

Identifying and Removing Outliers: The code uses a scatter plot to visualize the data and identify outliers in person_age, person_income, and person_emp_length. The identified outliers are then removed from the dataset using appropriate conditions.<br/>

Balancing the Dataset: The code checks the class distribution in the target variable (loan_status) and identifies that it is imbalanced. Further actions could be taken to balance the dataset if required (e.g., oversampling, undersampling, or using appropriate sampling techniques).<br/>

One-Hot Encoding: Categorical variables (person_home_ownership, loan_intent, loan_grade, and cb_person_default_on_file) are one-hot encoded to convert them into numerical features, which can be used by machine learning algorithms.<br/>

**Model Building and Evaluation**<br/>
The code builds several machine learning models for credit risk analysis:<br/>

k-Nearest Neighbors (KNN) Classifier: A KNN classifier with n_neighbors=4 is trained and evaluated using classification_report to measure precision, recall, F1-score, and accuracy.<br/>

Gaussian Naive Bayes: A Gaussian Naive Bayes classifier is trained and evaluated in a similar way.<br/>

Logistic Regression: A logistic regression classifier is trained and evaluated using the same metrics.<br/>

Support Vector Machine (SVM): A support vector machine classifier is trained and evaluated using the same metrics.<br/>

Each model's performance is reported using the classification_report function, which gives a detailed summary of precision, recall, F1-score, and support for both classes (0 and 1).
