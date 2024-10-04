![bdfe8ac2-c921-49f4-9504-a8483e7acef1](https://github.com/user-attachments/assets/995ff969-641d-48bb-9d76-b58374ac480b)
_________________________________________________________________________________________________________________________
# Car Insurance Claim Prediction

## Overview
This project aims to design a predictive solution to estimate the probability of claims on car insurance policies. By analyzing vehicle policy and safety data, we can forecast whether a customer is likely to file a claim within a 3 month timeframe.
_________________________________________________________________________________________________________________________
## Usage Examples
To run the application, use the following command:
```python
python app.py
```
## Installation Instructions
Ensure you have the necessary dependencies installed, which are listed in the requirements.txt file.
_________________________________________________________________________________________________________________________
## Project Timeline
![napkin-selection (6)](https://github.com/user-attachments/assets/b5352464-14aa-40a4-912d-934bcce72f9b)


1. Data Preprocessing
```python
def remove_outliers(data, column_name):
    """
    Remove outliers from a specific column of a dataframe using the IQR method.
    """
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter out the outliers by keeping only valid values
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
```
2. Data Analysis
  
     
![car_insurance_policy_claim](https://github.com/user-attachments/assets/36680642-f59a-4e1e-a60e-a355d1c0fe58)

3. Feature Engineering
```python
One Hot Encoding
data_en = pd.get_dummies(data, columns = ['fuel_type','steering_type','segment'], drop_first = True)

data_en['model'] = data_en['model'].apply(lambda x : x [1:]).astype(int) # removing first char & then converting into int
```
4. Training model
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
print(train_eval_model(log_reg))
```
5. Model Evaluation
```python
rom sklearn.metrics import roc_curve, auc
y_prob = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,8))
plt.show()
```
6. Model Deployment

![Screenshot 2024-10-04 121336](https://github.com/user-attachments/assets/e0bbc272-f055-448e-a5c3-9c405d29f476)
____________________________________________________________________________________________________________
## Technologies Used
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imblearn
- Streamlit
- Render
____________________________________________________________________________________________________________
## Contact:
For any questions or feedback, feel free to contact me at akashambure123@gmail.com

Happy Coding! 🚀
____________________________________________________________________________________________________________
















