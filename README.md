# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we created a model that can identify credit card customers that are most likely to churn. But the main objective to implement best coding practices.The completed project include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). 

## Files and data description
The structure of this project directory tree is displayed as follows:
```
.
├── churn_library.py
├── churn_notebook.ipynb
├── Guide.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── customer_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transaction_distribution.png
│   │   
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt

```
* Folders
  * `data`: Folder that contains the data in `csv` format
  *  `images`: Main folder
    * `eda`: Folder that is used to store the results of visualizations of numerical results
    * `results`: Folder that is used to store the training evaluation results
  * `logs`: Folder that stores the logs for the test results on the churn_library.py file
  * `models`: Folder is used to store model objects
  
* Files
  * `churn_library.py`: The churn_library.py is a library of functions to find customers who are likely to churn.
  * `churn_notebook.ipynb`: the churn_notebook.ipynb file containing the solution to identify credit card customers that are most likely to churn, but without implementing the engineering and software best practices.
  * `Guide.ipynb`: The Guide.ipynb is the starting point for the project
  * `churn_script_logging_and_tests.py`: The churn_script_logging_and_tests.py contain unit tests for the churn_library.py functions

## Running Files

**Clone the project**<br>
`git clone https://github.com/PhilippeMitch/Predict-Customer-Churn.git`<br>
This command will load the project into your personel computer

**Install the required libraries**<br>
`pip install -r requirements_py3.8.txt`<br>
With this command you will install all the required library.

**Run the workflow**<br>
`python churn_library.py`<br>

**Testing and Logging**<br>
`python churn_script_logging_and_tests.py`<br>
This command will test each of the functions and provide any errors to a file stored in the /logs folder.