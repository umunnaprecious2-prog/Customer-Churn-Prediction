
Project: Customer Churn Prediction

Dataset

Iranian Churn Dataset (UCI Machine Learning Repository)

The dataset contains customer information collected from an Iranian telecommunications company, including demographic characteristics, service usage, billing details, customer behavior, and churn status. It is designed for predicting whether a customer will discontinue the service.

# Business Problem
A telecommunications provider wants to reduce customer attrition by identifying subscribers who are likely to leave in the near future.
The objective is not only to create an accurate prediction model but also to help the business prioritize retention efforts for valuable customers before they decide to leave.
What to Do

1. Explore the dataset and identify patterns related to customer churn.
2. Analyze important variables such as:
   * Call failures
   * Subscription length
   * Customer value
   * Charge amount
   * Complaints
   * Age
   * Frequency of service usage
3. Handle missing values and prepare the dataset for modeling.
4. Encode categorical variables and normalize numerical features where necessary.
5. Build a baseline model using Logistic Regression.
6. Compare the baseline with more advanced models such as:
    - Random Forest
    - XGBoost
    - LightGBM
7. Evaluate the models using:
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Confusion Matrix

# Business Analysis

Estimate the financial impact of customer churn.

Develop a customer risk score using:

Risk Score = Churn Probability × Customer Value

Rank customers according to this score to identify those who should be contacted first through retention campaigns.

# Important Consideration

Discuss the impact of prediction errors.

False Positive: A customer is predicted to churn but remains with the company, resulting in unnecessary marketing costs.

False Negative: A customer who is actually going to churn is not identified, leading to customer loss and reduced revenue.

Recommend an appropriate classification threshold by considering business objectives instead of relying only on model accuracy.

# Final Deliverables

- Customer churn prediction model
- Churn probability report
- Ranked list of high-risk customers
- Customer risk scoring system
- Feature importance analysis
- Customer retention recommendations
- Dashboard showing churn insights