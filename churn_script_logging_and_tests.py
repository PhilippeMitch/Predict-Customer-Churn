'''
Unit test for each function in the churn_library.py

Author: Mitch
Date: 28 April 2023
'''
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
'''
    try:
        data = cls.import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
        return data
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(data_path):
    '''
    test perform eda function
    '''
    try:
        cls.perform_eda(data_path)
        # Try to open all the image file and assign True when it's exist
        image_lst = [
            "churn_distribution.png",
            "customer_age_distribution.png",
            "heatmap.png",
            "marital_status_distribution.png",
            "total_transaction_distribution.png",
        ]
        for image in image_lst:
            assert os.path.exists(f"images/eda/{image}")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: Could not found the {image} image")
        raise err


def test_encoder_helper(df_, category_lst):
    '''
    test encoder helper
    '''
    try:
        df_ = cls.encoder_helper(df_, category_lst)
        for category in category_lst:
            assert df_[f'{category}_Churn'].shape[0] > 0
        logging.info("Testing encoder_helper: SUCCESS")
        return df_
    except KeyError as err:
        logging.error("Testing encoder_helper: ERROR")
        raise err


def test_perform_feature_engineering(df_):
    '''
    test perform_feature_engineering
    '''
    try:
        train_feature, testing_feature, train_label, test_label = cls.perform_feature_engineering(
            df_)
        assert len(train_feature) == len(train_label)
        assert len(testing_feature) == len(test_label)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return train_feature, testing_feature, train_label, test_label
    except AssertionError as err:
        logging.info("Testing perform_feature_engineering: \
	       Feature and Target should have the same lenght")
        raise err


def test_train_models(train_feature, testing_feature, train_label, test_label):
    '''
    test train_models
    '''
    try:
        cls.train_models(
            train_feature,
            testing_feature,
            train_label,
            test_label)
        image_lst = [
            "feature_importances.png",
            "logistic_results.png",
            "rf_results.png",
            "roc_curve_result.png"
        ]
        for image in image_lst:
            assert os.path.exists(f"images/results/{image}")
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.info(
            f"Testing train_models: The {image} is not found")
        raise err

    try:
        model_lst = ["rfc_model.pkl", "logistic_model.pkl"]
        for model in model_lst:
            assert os.path.exists(f"models/{model}")
    except AssertionError as err:
        logging.info("Testing train_models: we could not found all the models")
        raise err


if __name__ == "__main__":
    DATA_PATH = "./data/bank_data.csv"
    CATEGORY_LST = [
        "Gender", "Education_Level", "Marital_Status",
        "Income_Category", "Card_Category"]
    DF = test_import(DATA_PATH)
    test_eda(DF)
    DF = test_encoder_helper(DF, CATEGORY_LST)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(DF)
    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
