import pandas as pd
import numpy as np
from itertools import combinations
import warnings
import re
import tensorflow.keras as keras
import tensorflow as tf
import os
import sys

# changing the LIME results dataset to a more useful format
def init_processing(df):
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
    # Reset the index to convert the index (current column names) to a regular column
    new_df = new_df.reset_index()
    # Rename the columns
    new_df.columns = new_df.iloc[0]
    # Display the new DataFrame
    new_df = new_df.drop(0)
    new_df = new_df.reset_index(drop=True)
    new_df.drop("Unnamed: 0", axis=1, inplace=True)
    new_df.set_index("exp number", inplace=True)
    return new_df

# Calculate the standard deviation as a step size
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

# Get the LIME important features, student features and step sizes for a specific student
def get_lime_results(features, lime_path, idx, mapping):
    
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
    y_orig=mapping[results["real value"][0]]

    return important_features, student_features, step_sizes, y_orig


def MC_LIME(important_features, student_features, step_sizes, average_values, loaded_model, y_orig, debug=False, threshold=0.5, max_group_size=3):
    features_to_change = {}
    # Try groups of different sizes (1,2,3,...,max_group_size)
    for group_size in range(1, max_group_size):
        # Generate all possible combinations of important features of current size
        if debug:
            print("\n\n\n___NEW GROUP SIZE___")
            print(group_size)
            print(features_to_change)

        for feature_group in combinations(important_features.index, group_size):
            modified_features = student_features.copy()
            
            #all_out_of_bounds = False
            out_of_bounds_features = []
            #result_changed=False
            #if debug:
                #print("___NEW COMBINATION___")
                #print(feature_group)
            
            while True:
                step_modified = False
                # do a step for each feature in the group
                for feature in feature_group:
                    if feature not in out_of_bounds_features:
                        step = step_sizes.loc[feature]
                        mean = average_values[feature]
                        original_value = student_features[feature]
                        
                        # Determine direction towards the mean
                        if original_value < mean:
                            step_direction = 1
                        else:
                            step_direction = -1
                        
                        new_value = modified_features[feature] + step * step_direction
                        #if debug:
                            #print(feature)
                            #print(new_value)
                        # Check if the new value is within bounds
                        if 0 <= new_value <= 1:
                            # Apply the change
                            modified_features[feature] = new_value
                            step_modified = True
                        else:
                            out_of_bounds_features.append(feature)
                
                
                if not step_modified:
                    break  # No more valid steps to apply, all remaining features are out of bounds
                
                # Predict using the modified features
                inputs = modified_features.values.reshape(1, -1)
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    prediction = loaded_model.predict(inputs)[0][0]
                sys.stdout = old_stdout
                prediction = 0 if prediction < threshold else 1
                # If the prediction is different, add this group of features to features_to_change
                # If not, the while loop will continue adding a next step to each feature
                if prediction != y_orig:
                    feature_changes = tuple(modified_features[feature] for feature in feature_group)
                    features_to_change[tuple(feature_group)] = feature_changes
                    break
                
        if features_to_change:
            break
    
    return features_to_change