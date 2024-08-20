import argparse
import pandas as pd
from MC_LIME import *
import json
import os

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
    
    # Check for the presence of the JSON file
    json_file_path = f"MC_LIME_temp_{course}.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            existing_keys = set(data.keys())
            # Remove any instances that are already present as keys in the JSON file
            instances = np.array([inst for inst in instances if str(inst) not in existing_keys])
    
    return instances

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--COURSE', type=str, required=True, help='Course name')
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("COURSE:", args.COURSE)

    course=args.COURSE

    valid_courses = ["villesafricaines_001", "geomatique_003", "dsp_001"]
    
    assert course in valid_courses, f"Invalid course name: {course}. Must be one of {valid_courses}"

    NUM_WEEKS=5

    # True if we want to get MC_LIME explanations
    MC_LIME_bool=True

    # CHANGE HERE THE FEATURE SET
    feature_set=f"features/{course}_features_{str(NUM_WEEKS)}_weeks.csv"
    features = pd.read_csv(feature_set)

    # GET LIME MODEL
    model_path = 'models/'
    model_name = model_path + "lstm_bi_" + course + "_" + str(NUM_WEEKS) + "_weeks"

    loaded_model = keras.models.load_model(model_name)
    config = loaded_model.get_config() # Config information about the model
    print(config["layers"][0]["config"]["batch_input_shape"]) # model shape

    instances = get_feature_indices(course, features)

    print("Instances: ", str(instances))
    lime_path=f"uniform_eq_results_ori_{str(NUM_WEEKS)}_weeks/LIME/{str(course)}/dataframes/"
    mapping={"pass" : 0, "fail" : 1}


    print("\n\n👩‍🚀🚀 Houston, we have liftoff! Generating MC-LIME results!\n\n")

    # Initialize or load existing MC_LIME_results
    json_file_path = f'MC_LIME_temp_{str(course)}.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            MC_LIME_results = json.load(f)
    else:
        MC_LIME_results = {}

    count=0
    for index in instances:
        print(index)
        important_features, student_features, step_sizes, y_orig = get_lime_results(features, lime_path, index, mapping)
        average_values = features[important_features.index.values].mean()

        MC_LIME_results[str(index)] = MC_LIME(important_features, student_features, step_sizes, average_values, loaded_model, y_orig, debug=True)

        MC_LIME_results[str(index)] = {str(inner_key): list(inner_value) for inner_key, inner_value in MC_LIME_results[str(index)].items()}

        # Save MC_LIME_results as a JSON file
        with open(json_file_path, 'w') as f:
            json.dump(MC_LIME_results, f)
        count+=1
        print("\n\n👩‍🚀🚀 Houston, we have generated: " + str(count) + " results!\n\n")

    print("\n\n👩‍🚀🚀 Houston, successful landing! MC-LIME results GENERATED!\n\n")


if __name__ == "__main__":
    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Call your main function
    main()