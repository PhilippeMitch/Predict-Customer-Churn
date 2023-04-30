# library doc string
'''
The churn_library.py is a library of functions to find customers who are likely to churn.

Author: Mitch
Date: 28 April 2023
'''

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'

def import_data(file_path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data = pd.read_csv(file_path)
    return data


def perform_eda(df_):
    '''
    perform eda on df_ and save figures to images folder
    input:
            df_: pandas dataframe

    output:
            None
    '''
	# Save the target distribution
    df_['Churn'] = df_['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20,10))
    df_['Churn'].hist()
    plt.savefig('images/eda/churn_distribution.png')
    # Save the Customer Age Distribution
    plt.figure(figsize=(20,10))
    df_['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_distribution.png')

    # Save the Marital Status Distribution
    plt.figure(figsize=(20,10))
    df_.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_distribution.png')

    # Save distributions of 'Total_Trans_Ct' and add a smooth curve
    # obtained using a kernel density estimate
    plt.figure(figsize=(20,10))
    sns.histplot(df_['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/total_transaction_distribution.png')

    # Save the heatmap
    plt.figure(figsize=(20,10))
    sns.heatmap(df_.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/eda/heatmap.png')
    plt.close()

def encoder_helper(data, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # gender encoded column
    gender_lst = []
    gender_groups = data.groupby(category_lst[0]).mean()['Churn']
    for val in data['Gender']:
        gender_lst.append(gender_groups.loc[val])
    data['Gender_Churn'] = gender_lst
    #education encoded column
    edu_lst = []
    edu_groups = data.groupby(category_lst[1]).mean()['Churn']
    for val in data['Education_Level']:
        edu_lst.append(edu_groups.loc[val])
    data['Education_Level_Churn'] = edu_lst
    #marital encoded column
    marital_lst = []
    marital_groups = data.groupby(category_lst[2]).mean()['Churn']
    for val in data['Marital_Status']:
        marital_lst.append(marital_groups.loc[val])
    data['Marital_Status_Churn'] = marital_lst
    #income encoded column
    income_lst = []
    income_groups = data.groupby(category_lst[3]).mean()['Churn']
    for val in data['Income_Category']:
        income_lst.append(income_groups.loc[val])
    data['Income_Category_Churn'] = income_lst
    #card encoded column
    card_lst = []
    card_groups = data.groupby(category_lst[4]).mean()['Churn']
    for val in data['Card_Category']:
        card_lst.append(card_groups.loc[val])
    data['Card_Category_Churn'] = card_lst
    return data

def perform_feature_engineering(df_):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              training_feature: X training data
              testing_feature: X testing data
              training_label: y training data
              test_label: y testing data
    '''
    # Columns that we want to keep
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    label = df_['Churn']
    feature = pd.DataFrame()
    feature[keep_cols] = df_[keep_cols]
    # train test split
    training_feature, testing_feature, training_label, test_label = train_test_split(
        feature, label, test_size= 0.3, random_state=42)
    
    return training_feature, testing_feature, training_label, test_label
       
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Save the classification report for the Random Forest
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')
    plt.close()
    # Save the classification report for the Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)), 
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'), {'fontsize': 10},
               fontproperties = 'monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.savefig('images/results/logistic_results.png')
    plt.close()
    
def feature_importance_plot(model, df_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            df_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [df_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(df_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(df_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + '/feature_importances.png')
    plt.close()

def train_models(training_feature, testing_feature, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              training_feature: X training data
              testing_feature: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Create a Random Forest Model
    rfc = RandomForestClassifier(random_state=42)
    # Create a Logistic Regression Model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # Declare some hyperparameter for the GridSearch
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    # Fine tune the Random Forest Model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(training_feature, y_train)
    # Train Logistic Regression Model
    lrc.fit(training_feature, y_train)
    # Evaluate the Random Forest Model base on the best estimator
    y_train_preds_rf = cv_rfc.best_estimator_.predict(training_feature)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(testing_feature)
    # Evaluate the LOgistic Regression model
    y_train_preds_lr = lrc.predict(training_feature)
    y_test_preds_lr = lrc.predict(testing_feature)
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    # Save the Roc Curve Result
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    #rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lrc, testing_feature, y_test)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('images/results/roc_curve_result.png')
    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


if __name__ == "__main__":
    FILE_PATH = "data/bank_data.csv"
    OUTPUT_PTH = "images/results"
    DF = import_data(FILE_PATH)
    perform_eda(DF)
    CATEGORY_LST = ["Gender", "Education_Level",
                    "Marital_Status", "Income_Category", "Card_Category"]
    DF = encoder_helper(DF, CATEGORY_LST)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DF)
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    RFC_MODEL = joblib.load('models/rfc_model.pkl')
    LR_MODEL = joblib.load('models/logistic_model.pkl')
    X_DATA = pd.concat([X_TRAIN, X_TEST])
    feature_importance_plot(RFC_MODEL, X_DATA, OUTPUT_PTH)
    Y_TRAIN_PREDS_RF = RFC_MODEL.predict(X_TRAIN)
    Y_TEST_PREDS_RF = RFC_MODEL.predict(X_TEST)
    Y_TRAIN_PREDS_LR = LR_MODEL.predict(X_TRAIN)
    Y_TEST_PREDS_LR = LR_MODEL.predict(X_TEST)
    classification_report_image(Y_TRAIN,
                                Y_TEST,
                                Y_TRAIN_PREDS_LR,
                                Y_TRAIN_PREDS_RF,
                                Y_TEST_PREDS_LR,
                                Y_TEST_PREDS_RF)
    