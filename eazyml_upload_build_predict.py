import json
import getopt
import sys
import os
from datetime import datetime
import eazyml as ez


AUTH_FILE = "authentication.json"
UPLOAD_FILE = "upload_data.json"
DT_TYPE_FILE = "data_type.json"
PREDICTIVE_MODEL_FEATURES_FILE = "predictive_model_feature_selected.json"
PREDICTIVE_MODEL_FILE = "predictive_model_performance_metrics.json"
AUGI_MODEL_FEATURES_FILE = "augi_insights_feature_selected.json"
AUGI_MODEL_FILE = "augi_insights.json"
PREDICT_RESULTS_FILE = "predictions.json"
EXPLANATIONS_FILE = "explanations.json"


def eazyml_upload(token, train_dataset_name, outcome, id_col="null",
                  discard_cols_list=[], impute="no", outlier="no",
                  prefix_name=""):
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
                "outlier": outlier,
                "discard": discard_cols_list,
                "outcome": outcome,
                "shuffle": "yes",
                "accelerate": "yes"
                }

    if not os.path.exists(train_dataset_name):
        print("Training data file doesn't exist - %s" % train_dataset_name)
        return None
    dump_file = prefix_name + "_" + UPLOAD_FILE
    time_start = datetime.now()
    print("Uploading the training data file ...")
    resp = ez.ez_load(token, train_dataset_name, options)
    if resp["success"] is True:
        print("The data file is uploaded successfully to EazyML: %s" %
              (train_dataset_name))
        dataset_id = resp["dataset_id"]
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + UPLOAD_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("The response is stored in %s" % (dump_file))
    else:
        print("Upload error: %s" % (resp["message"]))
        dataset_id = None
    ez_data_types = ez.ez_types(token, dataset_id)
    dt_json_obj = json.dumps(ez_data_types, indent=4)
    dump_file = prefix_name + "_" + DT_TYPE_FILE
    with open(dump_file, "w") as fw:
        fw.write(dt_json_obj)
    print("The inferred data types are stored in %s" % (dump_file))
    print("Upload time: " +
          "%.2f secs" % (datetime.now() - time_start).total_seconds())
    print("The reference identifier for the dataset " +
          "(dataset_id) is: %s" % (dataset_id))
    print("Likely next steps:")
    print("    python eazyml_upload_build_predict.py " +
          "--dataset_id %s --augi --no_clean_resources "
          "--prefix_name '%s' " % (dataset_id, prefix_name))
    print("    python eazyml_upload_build_predict.py " +
          "--dataset_id %s --predictive --no_clean_resources "
          "--prefix_name '%s' " % (dataset_id, prefix_name))
    return dataset_id


def eazyml_build_model(token, dataset_id, model_type,
                       nlp_options=None, date_time_column=None,
                       prefix_name="", valid_file=None, validate=False):
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
               "accelerate": "no"
              }
    if date_time_column is not None:
        options["date_time_column"] = date_time_column
        options["accelerate"] = "yes"

    print("Building " + model_type + " models ...")

    # Model initialization
    time_start = datetime.now()
    resp = ez.ez_init_model(token, dataset_id, options)
    if resp["success"] is True:
        print(model_type.title() + " models: initialized successfully")
    else:
        print("Model initialization error: %s" % (resp["message"]))
        return None
    print("Model initialization time: " +
          "%.2f secs" % (datetime.now() - time_start).total_seconds())
    model_id = resp["model_id"]
    if date_time_column is not None:
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + PREDICTIVE_MODEL_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("Performance metrics are stored in %s" % (dump_file))
        print("Model building time: " +
              "%.2f secs" % (datetime.now() - time_start).total_seconds())
        return model_id

    # derive text features if any
    time_start = datetime.now()
    if nlp_options is None:
        nlp_options = {
               "text_types": {'*': ["sentiments"]},
               "return_dataset": "no"
               }
    if type(nlp_options) == dict:
        resp = ez.ez_derive_text(token, model_id, nlp_options)
    if resp["success"] is True:
        print(model_type.title() + " models: text features " +
              "(sentiments) derived successfully")
        print("Text features derivation time: " +
              "%.2f secs" % (datetime.now() - time_start).total_seconds())
    elif "Text columns are not present" not in resp["message"]:
        print("Text features derivation error: %s" % (resp["message"]))

    # Feature selection
    time_start = datetime.now()
    resp = ez.ez_select_features(token, model_id)
    if resp["success"] is True:
        print(model_type.title() + " models: features selected successfully")
    else:
        print("Model feature selection error: %s" % (resp["message"]))
        return None

    if model_type == "predictive":
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + PREDICTIVE_MODEL_FEATURES_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("Selected features are stored " +
              "in %s" % (dump_file))
    else:
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + AUGI_MODEL_FEATURES_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("Selected features are stored " +
              "in %s" % (dump_file))

    print("Feature selection time: " +
          "%.2f secs" % (datetime.now() - time_start).total_seconds())

    # Build model
    time_start = datetime.now()
    resp = ez.ez_build_models(token, model_id)
    if resp["success"] is True:
        print(model_type.title() + " models built successfully")
    else:
        print("Model building error: %s" % (resp["message"]))
        return None
    if model_type == "predictive":
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + PREDICTIVE_MODEL_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("Performance metrics are stored in %s" % (dump_file))
        print("Model building time: " +
              "%.2f secs" % (datetime.now() - time_start).total_seconds())
    else:
        if validate:
            resp = ez.ez_validate(token, model_id, valid_file)
        json_obj = json.dumps(resp, indent=4)
        dump_file = prefix_name + "_" + AUGI_MODEL_FILE
        with open(dump_file, "w") as fw:
            fw.write(json_obj)
        print("Augmented-Intelligence insights are stored %s" % (dump_file))
        print("Model building time: " +
              "%.2f secs" % (datetime.now() - time_start).total_seconds())
        return model_id, resp
    print("The reference identifier for the model " +
          "(model_id) is: %s" % (resp["model_id"]))
    if model_type == "predictive":
        print("Likely next steps:")
        print(("    python eazyml_upload_build_predict.py " +
               "--model_id %s --predict_file <test_data_file> " +
               "--no_clean_resources " +
               "--prefix_name '%s'") % (model_id, prefix_name))
    return model_id


def eazyml_predict_dataset(token, model_id, predict_filename, prefix_name):
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
        print("Prediction data file doesn't exist - %s" % predict_filename)
        return None

    dump_file = prefix_name + "_" + PREDICT_RESULTS_FILE

    print("Uploading the prediction data file and making predictions ...")
    time_start = datetime.now()
    resp = ez.ez_predict(token, model_id, predict_filename)
    if resp["success"] is True:
        print("Predictions are ready")
    else:
        print("Prediction error: %s" % (resp["message"]))
        return None

    json_obj = json.dumps(resp, indent=4)
    dump_file = prefix_name + "_" + PREDICT_RESULTS_FILE
    with open(dump_file, "w") as fw:
        fw.write(json_obj)
    print("Predictions are stored in %s" % (dump_file))
    print("Prediction time: " +
          "%.2f secs" % (datetime.now() - time_start).total_seconds())
    print("The reference identifier for predictions " +
          "(prediction_dataset_id) is: %s" % (resp["prediction_dataset_id"]))
    print("Likely next steps:")
    print(("    python eazyml_upload_build_predict.py " +
           "--model_id %s --prediction_dataset_id %s " +
           "--explain_rec_nums <comma separated numbers> " +
           "--no_clean_resources --prefix_name '%s'")
          % (model_id, resp["prediction_dataset_id"], prefix_name))
    return resp["prediction_dataset_id"]


def eazyml_explain_points(token, model_id, prediction_dataset_id,
                          record_numbers, prefix_name):
    """
    Fetch explanations for the given record numbers

    Input:
        token: Authentication token
        model_id: model_id returned by predictive model
        prediction_dataset_id: Prediction id returned by predict function
        record_numbers: List of record numbers in prediction file

    """
    dump_file = prefix_name + "_" + EXPLANATIONS_FILE

    options = {"record_number": record_numbers}
    print("Executing Explainable-AI ...")
    time_start = datetime.now()
    resp = ez.ez_explain(token, model_id, prediction_dataset_id, options)
    if resp["success"] is True:
        print("Explanation/s is/are ready")
    else:
        print("Explanation error: %s" % (resp["message"]))
        return resp

    json_obj = json.dumps(resp, indent=4)
    dump_file = prefix_name + "_" + EXPLANATIONS_FILE
    with open(dump_file, "w") as fw:
        fw.write(json_obj)
    print("Explanations are stored in %s" % (dump_file))
    print("Explanations time: " +
          "%.2f secs" % (datetime.now() - time_start).total_seconds())
    return resp


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
        print("Authentication successful")
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


def clean_everything(token, dataset_id_list=[],
                     model_id_list=[],
                     prediction_dataset_id_list=[]):
    """
        Delete the dataset, model and preditions made if any
    """
    if prediction_dataset_id_list:
        for idx in prediction_dataset_id_list:
            if idx:
                resp = ez.ez_delete_test_datasets(token, idx)
                print(resp["message"])
    if model_id_list:
        for idx in model_id_list:
            if idx:
                resp = ez.ez_delete_models(token, idx)
                print(resp["message"])
    if dataset_id_list:
        for idx in dataset_id_list:
            if idx:
                resp = ez.ez_delete_datasets(token, idx)
                print(resp["message"])


def flow(username, api_key, config_file=None, prefix_name="",
         train_dataset_name=None, outcome=None, id_col="null",
         discard_col_list=[], impute="no", outlier="no",
         dataset_id=None, model_type=None, model_id=None,
         predict_filename=None, prediction_dataset_id=None,
         explain_record_numbers=[], clean_resources=True):
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
            print("Please authenticate to proceed")
            return

    cleanup_req = False
    # Set config file
    if config_file:
        print("Uploading the configuration file ...")
        resp = ez.ez_config(token, config_file)
        if resp["success"] is True:
            print("Configuration file is uploaded successfully")
        else:
            print("Configuration file upload error: %s" % (resp["message"]))
            return

    # upload data and get dataset_id
    if train_dataset_name:
        if not outcome:
            print("Please provide the outcome column name")
            return
        dataset_id = eazyml_upload(token, train_dataset_name,
                                   outcome, id_col, discard_col_list,
                                   impute, outlier, prefix_name)
        cleanup_req = True
        if not dataset_id:
            return

    # Build model
    if model_type and dataset_id:
        model_id = eazyml_build_model(token, dataset_id, model_type,
                                      prefix_name=prefix_name,
                                      valid_file=train_dataset_name)
        if not model_id:
            return
        cleanup_req = True

    # Predict
    if predict_filename and model_id:
        prediction_dataset_id = eazyml_predict_dataset(token, model_id,
                                                       predict_filename,
                                                       prefix_name)
        cleanup_req = True
        if not prediction_dataset_id:
            return

    # Explain points
    if explain_record_numbers and model_id and prediction_dataset_id:
        eazyml_explain_points(token, model_id, prediction_dataset_id,
                              explain_record_numbers, prefix_name)

    if clean_resources and cleanup_req:
        clean_everything(token, [dataset_id],
                         [model_id],
                         [prediction_dataset_id])
        print("Resources cleanup done!")


if __name__ == "__main__":
    args_list = sys.argv[1:]
    # Options
    options = "hu:p:g:x:" + \
              "f:o:i:c:ul" + \
              "d:m:va" + \
              "r:t:e:s"
    long_options = ["help", "username=", "api_key=", "config_file=",
                    "prefix_name=", "train_file=", "outcome=",
                    "id_col=", "discard_col_list=", "impute", "outlier",
                    "dataset_id=", "model_id=", "predictive", "augi",
                    "predict_file=", "prediction_dataset_id=",
                    "explain_rec_nums=", "no_clean_resources"]
    username = api_key = config_file = None
    impute = outlier = "no"
    train_file = outcome = id_col = discard_col_list = dataset_id = None
    model_id = model_type = None
    predict_filename = prediction_dataset_id = None
    explain_record_numbers = None
    clean_resources = True
    prefix_name = "EazyML"
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
            elif curr_arg in ("-x", "--prefix_name"):
                prefix_name = curr_val
            elif curr_arg in ("-f", "--train_file"):
                train_file = curr_val
            elif curr_arg in ("-o", "--outcome"):
                outcome = curr_val
            elif curr_arg in ("-i", "--id_col"):
                id_col = curr_val
            elif curr_arg in ("-c", "--discard_col_list"):
                discard_col_list = [x.strip() for x in curr_val.split(",")]
            elif curr_arg in ("-u", "--impute"):
                impute = "yes"
            elif curr_arg in ("-l", "--outlier"):
                outlier = "yes"
            elif curr_arg in ("-s", "--no_clean_resources"):
                clean_resources = False
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
                explain_record_numbers = [x.strip()
                                          for x in curr_val.split(",")]

        flow(username, api_key, config_file, prefix_name,
             train_file, outcome, id_col,
             discard_col_list, impute, outlier,
             dataset_id, model_type, model_id,
             predict_filename, prediction_dataset_id,
             explain_record_numbers, clean_resources)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
