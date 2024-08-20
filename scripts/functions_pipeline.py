import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
import re
import json
import os
import openai
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
import argparse
from dotenv import load_dotenv, find_dotenv
import warnings
import tensorflow.keras as keras
import tensorflow as tf
from itertools import combinations

# Other files and functions
from prompt_class import prompt_class
from chat_open_ai import chat_openai_model
from prompt_templates import TEMPLATES

manual_mapping = {
    'reg_periodicity_day_hour': 'regularity_periodicity_m1',
    "reg_peak_time_day_hour": "regularity_peak_dayhour",
    'number_of_sessions': 'number_sessions',
    'avg_time_sessions': 'time_sessions_mean',
    'total_time_sessions': 'time_sessions_sum',
    'std_time_sessions': 'time_sessions_std',
    'std_time_between_sessions': 'time_between_sessions_std',
    'total_clicks': 'total_clicks',
    'total_time_problem': 'time_in__problem_sum',
    'total_time_video': 'time_in__video_sum',
    "total_clicks_video": "total_clicks_Video",
    'total_clicks_video_load': 'total_clicks_Video.Load',
    'frequency_event_load': 'frequency_action_Video.Load',
    'avg_watched_weekly_prop': 'weekly_prop_watched_mean',
    'avg_replayed_weekly_prop': 'weekly_prop_replayed_mean',
    'content_alignment': 'content_alignment'
}

def get_feature_indices(course, features):
    # Load the feature indices
    index = pd.read_csv(f"../easy-fail/eq_week-chen_cui-{course}/feature_labels.csv")
    
    # Make the index a column called 'feature_index'
    index.reset_index(inplace=True)
    index.rename(columns={'index': 'feature_index'}, inplace=True)
    
    # Rename 'Unnamed: 0' to 'original_index'
    index.rename(columns={'Unnamed: 0': 'original_index'}, inplace=True)
    
    # Keep only 'feature_index' and 'original_index' columns
    index = index[['feature_index', 'original_index']]
    
    # Load the samples_id DataFrame
    samples_id = pd.read_csv(f"data/{course}_sample_100.csv")
    
    # Create an index mapping
    index_mapping = dict(zip(index['original_index'], index['feature_index']))
    
    # Map the 'original_index' in samples_id to 'feature_index'
    samples_id['feature_index'] = samples_id['original_index'].map(index_mapping)
    
    # Get the feature indices as a numpy array
    instances = samples_id['feature_index'].values
    
    # Ensure all instances indices are within the bounds of the features DataFrame
    assert np.all(instances < features.shape[0]), "Some indices are out of bounds"
    
    return instances

def convert_key(key):
    new_key = ''
    for char in key:
        if char.isupper():
            new_key += '_' + char.lower()
        else:
            new_key += char
    return new_key.lstrip('_')

def get_model(model_path, course, num_weeks):
    """
    Load a trained model given the course and number of weeks.
    """

    model_name = model_path + "lstm_bi_" + course + "_" + str(num_weeks) + "_weeks"

    loaded_model = keras.models.load_model(model_name)
    config = loaded_model.get_config() # Config information about the model
    print(config["layers"][0]["config"]["batch_input_shape"]) # model shape
    return loaded_model


def init_processing(df):
    """
    Process the LIME results dataset to a more useful format.
    """

    regexp=re.compile(r' < .* <= ')
    for index, row in df.iterrows():
        s=row['Unnamed: 0']
        if regexp.search(s):
            s=s.split(' < ')[1].split(' <= ')[0]
        else:
            s = s.split(' <=')[0]
            s = s.split(' <')[0]
            s = s.split(' >')[0]
        row['Unnamed: 0']=s
    new_df = df.T
    new_df = new_df.reset_index()     # Reset the index to convert the index (current column names) to a regular column

    new_df.columns = new_df.iloc[0]  # Rename columns

    new_df = new_df.drop(0)
    new_df = new_df.reset_index(drop=True)
    new_df.drop("Unnamed: 0", axis=1, inplace=True)
    new_df.set_index("exp number", inplace=True)
    return new_df


def calculate_std_dev_step_sizes(df, fraction=1.0):
    """
    Calculates step sizes for each numerical column in a DataFrame based on the standard deviation.

    Parameters:
    - df: pandas DataFrame with numerical columns.
    - fraction: Fraction of the standard deviation to use as the step size.

    Returns:
    - A pandas Series where the index corresponds to column names and values to the step sizes.
    """
    step_sizes = {}
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        std_dev = df[column].std()
        step_size = std_dev * fraction
        step_sizes[column] = step_size
    
    return pd.Series(step_sizes)

def predict(model, features):
    """
    Predict the outcome of a student based on the model and features (inference).
    """
    threshold = 0.5
    pred_to_string={0: "pass", 1: "fail"}
    input = features.values.reshape(1, -1)
    prediction = model.predict(input)[0][0]
    pred_class = 0 if prediction < threshold else 1
    confidence = 1-abs(pred_class - prediction)
    confidence*=100
    pred_class = pred_to_string[pred_class]
    return pred_class, confidence


def get_lime_results(loaded_model, features, lime_path, idx, mapping):
    """
    Get the LIME results for a specific student and process them.
    """
    results=pd.read_csv(lime_path + str(idx) + ".csv")
    results=init_processing(results)

    top_columns=results.columns.values
    top_columns = top_columns[top_columns != 'real value']
    selected_features=features[top_columns]
    
    step_sizes = calculate_std_dev_step_sizes(selected_features)
    
    # Get the important features for this student based on LIME
    id_number=results.iloc[0].name
    important_features=results.T[1:]

    # Get the student's features
    important_features.index.name="feature"
    important_features.columns=["LIME_score"]
    student_features=features.iloc[int(id_number)]

    # Count the number of features with zero step size
    zero_count = (step_sizes == 0).sum()
    assert zero_count == 0, "There are features with zero step size. Please check the data."


    # Get the predicted value
    y_pred, confidence = predict(loaded_model, student_features)

    # Get the real value
    y_orig=mapping[results["real value"][0]]

    return important_features, student_features, step_sizes, y_pred, confidence

# Function to convert dictionary to formatted string
def get_features_description(features):
    with open('features/features_description.json', 'r') as f:
        features_desc_json = json.load(f)

    # Convert all dictionary keys
    converted_dict = {convert_key(key): value for key, value in features_desc_json.items()}

    features_no_week = [re.sub(r'_InWeek\d+', '', s) for s in features.columns.values]
    features_no_week=set(features_no_week)

    # Filtered dictionary with only matching keys
    features_descriptions_dict = {key: converted_dict[key] for key in converted_dict if key in features_no_week}

    # Convert and add mapped elements to matching_dict
    for dict_key, set_key in manual_mapping.items():
        if dict_key in converted_dict:
            features_descriptions_dict[set_key] = converted_dict[dict_key]
    return '\n'.join(f"{key}:{value}" for key, value in features_descriptions_dict.items()) + ','

def define_post_hoc_description(LIME_bool, MC_LIME_bool, CEM_bool): 
    if LIME_bool and MC_LIME_bool and CEM_bool:
        post_hoc_description = "Ensemble: LIME, MC-LIME and CEM"
    if LIME_bool:
        post_hoc_description = TEMPLATES["post_hoc_description"]["LIME"]
    elif MC_LIME_bool:
        post_hoc_description =  TEMPLATES["post_hoc_description"]["MC_LIME"]
    elif CEM_bool:
        post_hoc_description = TEMPLATES["post_hoc_description"]["CEM"]
    return post_hoc_description

def get_data(course, loaded_model, idx, instances, features, lime_path, cem_path, MC_LIME_bool, LIME_bool, CEM_bool, average_class=False):
        '''
        Retrieves and processes data related to a specific student.

            Parameters:
            - idx (int): The unique identifier for the student.
            - features (DataFrame): A DataFrame containing feature scores for students for that specific number of weeks.

            Returns:
            - str: A formatted string summarizing the processed data.
        '''
        mapping={"pass" : 0, "fail" : 1}
        inverse_mapping = {0: "pass", 1: "fail"}
        data_string = ""
        important_features, student_features, step_sizes, y_pred, confidence = get_lime_results(loaded_model, features, lime_path, idx, mapping)
        
        average_values = features[important_features.index.values].mean()
            
        # Create dict for important features values
        selected_features_dict = student_features[important_features.index.values]
        selected_features_dict = selected_features_dict.to_dict()

        data_string += f"MODEL PREDICTION: {y_pred}, with {confidence:.6f}% of confidence.\n"
        # Add Lime scores to the string if LIME_bool is True
        if LIME_bool:
            # Create dict for important features scores
            lime_scores_dict = important_features.to_dict(orient='dict')
            lime_scores_dict = lime_scores_dict['LIME_score']
            data_string += f"\nFEATURE IMPORTANCES\nThese are the features found important by LIME: \n{lime_scores_dict}\n"

        if MC_LIME_bool:
            # Read the JSON file
            with open(f'data/MC_LIME_results_{course}.json', 'r') as file:
                MC_LIME_results = json.load(file)
            # read the file with the index bla bla
            string_MC_LIME = str(MC_LIME_results[str(idx)])
            data_string += f"\nFEATURE IMPORTANCES\nThese are the set of features found important by Minimal-Counterfactual LIME: \n{string_MC_LIME}\n"
        
        if CEM_bool:
            if idx==5192 and course=="dsp_001":
                # Load .npy file containing multiple arrays
                path_index= cem_path + f"changes_pn_{idx}.npy"
                array = np.load(path_index, allow_pickle=True)
                cem_df = pd.DataFrame(array)
                cem_df = cem_df.T
                cem_df['feature_name'] = features.columns.values
                cem_df.rename(columns={0: 'change'}, inplace=True)
                cem_df.sort_values(ascending=False, by='change', inplace=True)
                string_CEM = ""
                for _, row in cem_df[:10].iterrows():
                    string_CEM += f"{row['feature_name']} - {row['change']:.6f}\n"
                data_string += f"\nFEATURE IMPORTANCES\nThese are the features found important by CEM: \n{string_CEM}\n"
                selected_features_dict=student_features[cem_df[:10]["feature_name"]]
                selected_features_dict = selected_features_dict.to_dict()
            else:
                cem_instances = instances
                # Load .npy file containing multiple arrays
                path_index= cem_path + f"changes_pn.npy"
                array = np.load(path_index, allow_pickle=True)
                cem_df = pd.DataFrame(array)
                cem_df = cem_df.T

                cem_df.columns = cem_instances
                cem_df['feature_name'] = features.columns.values
                idx_cem_df= cem_df[[idx, 'feature_name']].copy()
                idx_cem_df.sort_values(ascending=False, by=idx, inplace=True)
                string_CEM = ""
                for _, row in idx_cem_df[:10].iterrows():
                    string_CEM += f"{row['feature_name']} - {row[idx]:.6f}\n"
                selected_features_dict=student_features[idx_cem_df[:10]["feature_name"]]
                selected_features_dict = selected_features_dict.to_dict()
                data_string += f"\nFEATURE IMPORTANCES\nThese are the features found important by CEM: \n{string_CEM}\n"

        # Add feature values
        if CEM_bool:
            data_string += f"FEATURE VALUES\nThe relevant feature values found by CEM for the student are included below: {selected_features_dict}\n"
        else:
            data_string += f"FEATURE VALUES\nThe relevant feature values found by LIME for the student are included below: {selected_features_dict}\n"

        if average_class:
            data_string += f"\nAVERAGE VALUES\nAverage feature values of the class: \n{average_values}"

        return data_string

def langchain_prompt_to_str(prompt):
    return str(prompt[0]).split("content=")[1]