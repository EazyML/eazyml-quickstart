import json
import getopt
import sys
import os
import eazyml as ez


AUTH_FILE = "authentication.json"
UPLOAD_FILE = "upload_data.json"
PREDICTIVE_MODEL_FILE = "predictive_model_performance_metrics.json"
AUGI_MODEL_FILE = "augi_insights.json"
PREDICT_RESULTS_FILE = "predictions.json"
EXPLANATIONS_FILE = "explanations.json"


def eazyml_upload(token, train_dataset_name, outcome, id_col="null",
                  discard_cols_list=[], impute="no"):
    """
    Upload data on EazyML

    Input:
        token: Authentication token
        train_dataset_name: Train dataset name
        outcome: Outcome column in train dataset
        id_col: ID column of train data. If no ID column in train data,
                default is "null"
        discard_cols_list: List of columns to be discarded

    Return:
        Returns the dataset_id
    """
    options = {
                "id": id_col,
                "impute": impute,
                "outlier": "yes",
                "discard": discard_cols_list,
                "outcome": outcome,
                "shuffle": "yes",
                "accelerate": "yes"
                }

    if not os.path.exists(train_dataset_name):
        print("Train file doesn't exist - %s" % train_dataset_name)
        return None

    print("Uploading dataset ...")
    resp = ez.ez_load(token, train_dataset_name, options)
    if resp["success"] is True:
        print("Data uploaded successfully on EazyML: %s" %
              (train_dataset_name))
        dataset_id = resp["dataset_id"]
        json_obj = json.dumps(resp, indent=4)
        with open(UPLOAD_FILE, "w") as fw:
            fw.write(json_obj)
        print("The response is stored in %s" % (UPLOAD_FILE))
    else:
        print("Upload error: %s" % (resp["message"]))
        dataset_id = None

    print("The reference identifier for the dataset (dataset_id) is: %s" % (dataset_id))
    return dataset_id


def eazyml_build_model(token, dataset_id, model_type):
    """
    Build predictive or augmented intelligence models

    Input:
        token: Authentication token
        dataset_id: dataset_id returned by upload function
        model_type: predictive or augmented intelligence

    Return:
        Returns the model_id
    """
    options = {
               "model_type": model_type,
               "accelerate": "yes"
              }
    print("Building " + model_type + " models ...")
    resp = ez.ez_init_model(token, dataset_id, options)
    if resp["success"] is True:
        print(model_type.title() + " models built successfully.")
        print("The reference identifier for the model (model_id) is: %s" % (resp["model_id"]))
    else:
        print("Model build error: %s" % (resp["message"]))
        return None
    if model_type == "predictive":
        json_obj = json.dumps(resp, indent=4)
        with open(PREDICTIVE_MODEL_FILE, "w") as fw:
            fw.write(json_obj)
        print("Performance metrics are stored in %s" % (PREDICTIVE_MODEL_FILE))
    else:
        json_obj = json.dumps(resp["insights"], indent=4)
        with open(AUGI_MODEL_FILE, "w") as fw:
            fw.write(json_obj)
        print("Augmented intelligence insights are stored %s" % (AUGI_MODEL_FILE))
        return None
    return resp["model_id"]


def eazyml_predict_dataset(token, model_id, predict_filename):
    """
    Generate predictions on the predictive model built

    Input:
        token: Authentication token
        model_id: model_id returned by predictive model
        predict_filename: Prediction file name

     Return:
        Returns the test_dataset_id
    """
    if not os.path.exists(predict_filename):
        print("Predict file doesn't exist - %s" % predict_filename)
        return None

    print("Uploading the prediction dataset and making predictions ...")
    resp = ez.ez_predict(token, model_id, predict_filename)
    if resp["success"] is True:
        print("Predictions are ready.")
        print("The reference identifier for predictions " + \
              "(prediction_dataset_id) is: %s" % (resp["prediction_dataset_id"]))
    else:
        print("Prediction error: %s" % (resp["message"]))
        return None

    json_obj = json.dumps(resp, indent=4)
    with open(PREDICT_RESULTS_FILE, "w") as fw:
        fw.write(json_obj)
    print("Predictions are stored in %s" % (PREDICT_RESULTS_FILE))
    return resp["prediction_dataset_id"]


def eazyml_explain_points(token, model_id, prediction_dataset_id,
                          record_numbers):
    """
    Fetch explanations for the given record numbers

    Input:
        token: Authentication token
        model_id: model_id returned by predictive model
        prediction_dataset_id: Prediction id returned by predict function
        record_numbers: List of record numbers in prediction file
        
    """
    options = {"record_number": record_numbers}
    print("Executing Explainable-AI ...")
    resp = ez.ez_explain(token, model_id, prediction_dataset_id, options)
    if resp["success"] is True:
        print("Explanations is/are ready!!")
    else:
        print("Explanation error: %s" % (resp["message"]))
        return

    json_obj = json.dumps(resp, indent=4)
    with open(EXPLANATIONS_FILE, "w") as fw:
        fw.write(json_obj)
    print("Explanations are stored in %s" % (EXPLANATIONS_FILE))


def eazyml_auth(username, api_key, store_info=False):
    """
    Authenticate and store auth info in a file for future use

    Input:
        username: Email Id or username provided
        api_key: Api Key downloaded from UI
        store_info: Flag to store info for future use

    Return:
        Return authentication token used for sucessive calls to EazyML
    """
    resp = ez.ez_auth(username, api_key=api_key)
    if resp["success"] is True:
        print("Authentication successful.")
        if store_info:
            content = {"username": username,
                       "api_key": api_key}
            json_obj = json.dumps(content, indent=4)
            with open(AUTH_FILE, "w") as fw:
                fw.write(json_obj)
            print("Authentication information is stored in %s" % (AUTH_FILE))
    else:
        print("Authentication error: %s" % (resp["message"]))
        return None
    return resp["token"]


def flow(username, api_key, config_file=None,
         train_dataset_name=None, outcome=None, id_col="null",
         discard_col_list=[], impute="no", dataset_id=None,
         model_type=None, model_id=None,
         predict_filename=None, prediction_dataset_id=None,
         explain_record_numbers=[]):
    """
    Run the EazyML operations based on the input

    """
    # Get Authentication token
    token = None
    if username and api_key:
        token = eazyml_auth(username, api_key, store_info=True)
    else:
        if os.path.exists(AUTH_FILE):
            auth_info = json.load(open(AUTH_FILE, "r"))
            token = eazyml_auth(auth_info["username"],
                                auth_info["api_key"])
        else:
            print("Please authenticate to proceed.")
            return

    #Set config file
    if config_file:
        print("Uploading config file ...")
        resp = ez.ez_config(token, config_file)
        if resp["success"] is True:
            print("Config file uploaded and set successfully.")
        else:
            print("Config error: %s" % (resp["message"]))
            return
        
    # upload data and get dataset_id
    if train_dataset_name:
        if not outcome:
            print("Please provide outcome column name.")
            return
        dataset_id = eazyml_upload(token, train_dataset_name,
                                   outcome, id_col, discard_col_list, impute)
        if not dataset_id:
            return

    # Build model
    if model_type and dataset_id:
        model_id = eazyml_build_model(token, dataset_id, model_type)
        if not model_id:
            return

    # Predict
    if predict_filename and model_id:
        prediction_dataset_id = eazyml_predict_dataset(token, model_id,
                                                       predict_filename)
        if not prediction_dataset_id:
            return
    # Explain points
    if explain_record_numbers and model_id and prediction_dataset_id:
        eazyml_explain_points(token, model_id, prediction_dataset_id,
                              explain_record_numbers)


if __name__ == "__main__":
    args_list = sys.argv[1:]
    # Options
    options = "hu:p:g:" + \
              "f:o:i:c:u" + \
              "d:m:va" + \
              "r:t:e:"
    long_options = ["help", "username=", "api_key=", "config_file=",
                    "train_file=", "outcome=", "id_col=", "discard_col_list=", "impute",
                    "dataset_id=", "model_id=", "predictive", "augi",
                    "predict_file=", "prediction_dataset_id=",
                    "explain_rec_nums="]
    username = api_key = config_file = None
    impute = "no"
    train_file = outcome = id_col = discard_col_list = dataset_id = None
    model_id = model_type = None
    predict_filename = prediction_dataset_id = None
    explain_record_numbers = None
    try:
        # Parsing argument
        arguments, values = getopt.getopt(args_list, options, long_options)

        # checking each argument
        for curr_arg, curr_val in arguments:
            print(curr_arg, curr_val)
            if curr_arg in ("-h", "--help"):
                print("".join(open("help.txt", "r").readlines()))
                exit()
            elif curr_arg in ("-u", "--username"):
                username = curr_val
            elif curr_arg in ("-p", "--api_key"):
                api_key = curr_val
            elif curr_arg in ("-g", "--config_file"):
                config_file = curr_val
            elif curr_arg in ("-f", "--train_file"):
                train_file = curr_val
            elif curr_arg in ("-o", "--outcome"):
                outcome = curr_val
            elif curr_arg in ("-i", "--id_col"):
                id_col = curr_val
            elif curr_arg in ("-c", "--discard_col_list"):
                discard_col_list = curr_val.split(",")
            elif curr_arg in ("-u", "--impute"):
                impute = "yes"
            elif curr_arg in ("-d", "--dataset_id"):
                dataset_id = curr_val
            elif curr_arg in ("-m", "--model_id"):
                model_id = curr_val
            elif curr_arg in ("-v", "--predictive"):
                model_type = "predictive"
            elif curr_arg in ("-a", "--augi"):
                model_type = "augmented intelligence"
            elif curr_arg in ("-r", "--predict_file"):
                predict_filename = curr_val
            elif curr_arg in ("-t", "--prediction_dataset_id"):
                prediction_dataset_id = curr_val
            elif curr_arg in ("-e", "--explain_rec_nums"):
                explain_record_numbers = curr_val.split(",")

        flow(username, api_key, config_file,
             train_file, outcome, id_col,
             discard_col_list, impute, dataset_id,
             model_type, model_id,
             predict_filename, prediction_dataset_id,
             explain_record_numbers)
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))
