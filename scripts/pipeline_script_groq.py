import pandas as pd
from MC_LIME import *
from langchain.prompts import ChatPromptTemplate
import re
from chat_groq import chat_groq_model

# External functions
from functions_pipeline import *
from prompt_templates import TEMPLATES
import warnings
from prompt_class import prompt_class
import random

from IPython.display import display, Markdown, Latex
import os
import datetime
import time
from tqdm import tqdm

def display_prompt(prompting_strategies, strategy, prompt_type):
    # Determine the column to display based on prompt_type
    if prompt_type == "prompt1":
        column_name = "prompt"
    elif prompt_type == "prompt2":
        column_name = "presentation_instruction"
    else:
        raise ValueError("Invalid prompt_type. Choose 'prompt1' or 'prompt2'.")

    # Filter the dataframe to get the specific strategy
    selected_prompt = prompting_strategies[prompting_strategies["name"] == strategy][column_name].values[0]
    selected_prompt = selected_prompt[0].content
    
    # Display the prompt using Markdown
    display(Markdown(selected_prompt))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--COURSE', type=str, required=True, help='Course name')
    parser.add_argument('--MODEL', type=str, required=True, help='Model name')
    parser.add_argument('--KEY', type=int, required=True, help='API key number') 
    return parser.parse_args()

def call_groq_model_with_retry(main_prompt, presentation_prompt, model_name, conversation_template, key=None):
    retry_count = 0  # Initialize the retry counter
    
    while True:
        try:
            # Call your function to interact with GROQ
            llm_class = chat_groq_model(model_name, 0.5, key)
            conversation = llm_class.define_chat_model(conversation_template)
            response1 = conversation.predict(input=main_prompt)
            response2 = conversation.predict(input=presentation_prompt)
            return response1, response2
        except Exception as e:
            retry_count += 1  # Increment the retry counter
            print(f"Rate limit reached. Attempt {retry_count}/5. Sleeping for 65 seconds.")
            time.sleep(65)
            
            # Check if we've retried more than 5 times
            if retry_count > 5:
                print("Maximum retry attempts reached. Returning empty responses.")
                return "", ""  # Return empty strings

def main():
    args = parse_arguments()
    print("COURSE:", args.COURSE)
    print("MODEL:", args.MODEL)
    print("KEY:", args.KEY)
    
    course=args.COURSE
    model_name=args.MODEL
    key=args.KEY

    valid_courses = ["villesafricaines_001", "geomatique_003", "dsp_001"]
    
    assert course in valid_courses, f"Invalid course name: {course}. Must be one of {valid_courses}"

    NUM_WEEKS=5

    LIME_bool=True
    MC_LIME_bool=False
    CEM_bool=False
    average_class=False

    # CHANGE HERE THE FEATURE SET
    feature_set=f"features/{course}_features_{str(NUM_WEEKS)}_weeks.csv"
    features = pd.read_csv(feature_set)
    features.shape

    # GET PREDICTION MODEL
    model_path = 'models/'

    loaded_model=get_model(model_path, course, NUM_WEEKS)

    indices = get_feature_indices(course, features)

    if course == "dsp_001":
        indices = indices[indices != 44]

    # Retrieve data for student idx
    lime_path=f"uniform_eq_results_ori_{str(NUM_WEEKS)}_weeks/LIME/{course}/dataframes/"
    cem_path=f"uniform_eq_results_ori_5_weeks/CEM/{course}/"
    mapping={"Pass" : 0, "Fail" : 1}

    ############################################################ TESTING WITH ONE INDEX
    # Change here the student index to test
    idx = random.choice(indices)

    print(f"\n\nRandomly selected index: {idx}")

    template=TEMPLATES["template"]
    goal_definition = TEMPLATES["goal_definition"]
    features_description = get_features_description(features)
    model_description = TEMPLATES["model_description"]
    post_hoc_description = define_post_hoc_description(LIME_bool, MC_LIME_bool, CEM_bool)
    format_instructions = TEMPLATES["format_instructions"]
    course_description = TEMPLATES[f"course_description_{course}"]
    data_string = get_data(course, loaded_model, idx, indices, features, lime_path, cem_path, MC_LIME_bool, LIME_bool, CEM_bool, average_class)
    conversation_template = TEMPLATES["conversation_template"]

    presentation_examples = TEMPLATES["presentation_examples"]
    presentation_template = TEMPLATES["presentation_template"]

    print(data_string)

    instructions_dataset=pd.read_csv("prompting_strategies.csv")
    instructions_dataset=instructions_dataset[['name', 'Communication Prompt', 'LLM-generated prompt']]
    instructions_dataset = instructions_dataset[instructions_dataset['name'] != 'hesslow_contrastive']
    instructions_dataset = instructions_dataset.rename(columns={'Communication Prompt': 'instruction'})
    instructions_dataset = instructions_dataset.rename(columns={'LLM-generated prompt': 'presentation_instruction'})

    prompt_object=prompt_class(template, instructions_dataset)

    # Preparing explanation selection template
    prompt_object.context_partial_prompt( 
        data_string,
        features_description, 
        goal_definition, 
        model_description, 
        post_hoc_description, 
        course_description,
        )
    
    # Preparing presentation template
    prompt_object.add_explanation_template(presentation_template)
    prompt_object.presentation_context_partial_prompt(course_description)

    # Generating prompts for each strategy
    prompt_object.create_prompts_list(instructions_dataset)

    ################################################### TESTING
    strategy = 'relevance_selection'
    display_prompt(prompt_object.prompting_strategies, strategy, "prompt1")

    test_prompt=prompt_object.prompting_strategies["prompt"][2]

    # DEFINE LANGCHAIN MODEL
    llm_class=chat_groq_model(model_name, 0.5)
    conversation=llm_class.define_chat_model(conversation_template)

    # Here we call the model with the main prompt
    result1=conversation.predict(input=test_prompt)

    test_presentation=prompt_object.prompting_strategies["presentation_prompt"][2]

    # Here we call the model with the presentation prompt
    result2=conversation.predict(input=test_presentation)

    print("Testing done! ✅")

    ################################################### PIPELINE
    
    print("\n\nPIPELINE LAUNCH! 🚀")

    with open(f'data/MC_LIME_results_{course}.json', 'r') as file:
        MC_LIME_results = json.load(file)

    no_MCLIME = [int(idx) for idx, result in MC_LIME_results.items() if not result]

    # MANY CALLS TO THE OPENAI API, BE CAREFUL WITH THE PRICE

    # Columns for the results DataFrame
    columns = ['student', 'explainer', 'strategy', 'first_prompt', 'second_prompt', 'response1', 'response2']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Check for an existing temporary file
    temp_file_path = f"data/{model_name}_answers_{course}_temp_{timestamp}.csv"
    if os.path.exists(temp_file_path):
        results_temp = pd.read_csv(temp_file_path)
        processed_indices = results_temp['student'].unique().tolist()
        indices = [idx for idx in indices if idx not in processed_indices]
    else:
        results_temp = pd.DataFrame(columns=columns)

    explainer_list = {
        "LIME": {"MC_LIME_bool": False, "LIME_bool": True, "CEM_bool": False},
        "MC-LIME": {"MC_LIME_bool": True, "LIME_bool": False, "CEM_bool": False},
        "CEM": {"MC_LIME_bool": False, "LIME_bool": False, "CEM_bool": True}
    }

    try:
        for idx in tqdm(indices, desc="Processing indices", total=len(indices)):
            for explainer in explainer_list:
                MC_LIME_bool = explainer_list[explainer]["MC_LIME_bool"]
                LIME_bool = explainer_list[explainer]["LIME_bool"]
                CEM_bool = explainer_list[explainer]["CEM_bool"]

                # Check if the student has a MCLIME explanation, if not, we only use LIME
                if idx in no_MCLIME and explainer == "MC-LIME":
                    MC_LIME_bool = False
                    LIME_bool = True
                    CEM_bool = False

                data_string = get_data(course, loaded_model, idx, indices, features, lime_path, cem_path, MC_LIME_bool, LIME_bool, CEM_bool)
                post_hoc_description = define_post_hoc_description(LIME_bool, MC_LIME_bool, CEM_bool)
                context_string = prompt_object.context_partial_prompt(
                    data_string,
                    features_description,
                    goal_definition,
                    model_description,
                    post_hoc_description,
                    course_description,
                )
                prompt_object.create_prompts_list(instructions_dataset)

                for index, row in prompt_object.prompting_strategies.iterrows():
                    print(f"Student: {idx}, Explainer: {explainer}, Strategy: {row['name']}\n")
                    main_prompt = row['prompt']
                    presentation_prompt = row['presentation_prompt']
                    
                    # Call the function with retry logic
                    response1, response2 = call_groq_model_with_retry(main_prompt, presentation_prompt, model_name, conversation_template, key)

                    # Append the results to the DataFrame
                    new_row = {
                        'student': idx,
                        'explainer': explainer,
                        'strategy': row['name'],
                        'first_prompt': main_prompt[0].content,
                        'second_prompt': presentation_prompt[0].content,
                        'response1': response1,
                        'response2': response2,
                    }

                    results_temp = pd.concat([results_temp, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        results_temp.to_csv(temp_file_path, index=False)
        raise

    
    print("\n\nPIPELINE FINISHED! ✅")

    file_path = f"data/{model_name}_answers_{course}_completed.csv"
    if os.path.exists(file_path):
        results_df = pd.read_csv(file_path)
    else:
        columns = ['student', 'explainer', 'strategy', 'first_prompt', 'second_prompt', 'response1', 'response2']
        results_df = pd.DataFrame(columns=columns)

    # Append the rows of results_temp to results_df using pd.concat
    results_df = pd.concat([results_df, results_temp], ignore_index=True)

    results_df.to_csv(file_path, index=False)




if __name__ == "__main__":
    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Call your main function
    main()