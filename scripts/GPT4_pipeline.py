import pandas as pd
from MC_LIME import *
import os
import openai
from dotenv import load_dotenv, find_dotenv
import warnings

# Other files and functions
from prompt_class import prompt_class
from chat_open_ai import chat_openai_model
from prompt_templates import TEMPLATES
from functions_pipeline import *
"""
This script is generating given the parameters A SINGLE ANSWER from the GPT-4 model from a random instruction strategy.
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--INDEX", type=int, default=None, help="Index of the student")
    parser.add_argument("--NUM_WEEKS", type=int, default=5, help="Number of weeks")
    parser.add_argument("--LIME", action="store_true", help="Boolean for LIME")
    parser.add_argument("--MC_LIME", action="store_true", help="Boolean for MC LIME")
    parser.add_argument("--CEM", action="store_true", help="Boolean for CEM")
    parser.add_argument("--average_class", action="store_true", help="Boolean for average class")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("INDEX: ", args.INDEX)
    print("NUM_WEEKS:", args.NUM_WEEKS)
    print("LIME_bool:", args.LIME)
    print("MC_LIME_bool:", args.MC_LIME)
    print("CEM_bool:", args.CEM)
    print("average_class:", args.average_class)

    NUM_WEEKS = args.NUM_WEEKS
    LIME_bool = args.LIME
    MC_LIME_bool = args.MC_LIME
    CEM_bool = args.CEM
    average_class = args.average_class

    # GET PREDICTION MODEL
    model_path = 'models/'
    course='dsp_001'
    loaded_model=get_model(model_path, course, NUM_WEEKS)

    # CHANGE HERE THE FEATURE SET
    feature_set=f"features/feature_set_{str(NUM_WEEKS)}_weeks.csv"
    features = pd.read_csv(feature_set)
    features.shape

    # GET THE INDICES OF THE 20 STUDENTS SELECTED
    mapping=pd.read_csv("data/user_id_mapping-dsp_001.csv", index_col=0)
    students_ids=pd.read_csv("data/representative_samples.csv")["user_id"]
    indices = mapping.index[mapping['user_id'].isin(students_ids)].tolist()

    # Retrieve data for student idx
    lime_path=f"uniform_eq_results_ori_{str(NUM_WEEKS)}_weeks/LIME/dsp_001/dataframes/"
    cem_path="uniform_eq_results_ori_5_weeks/CEM/dsp_001/"

    # Change here the student index to test
    idx=230

    # GET THE TEMPLATE VARIABLES
    template=TEMPLATES["template"]
    goal_definition = TEMPLATES["goal_definition"]
    model_description = TEMPLATES["model_description"]
    format_instructions = TEMPLATES["format_instructions"]
    course_description = TEMPLATES["course_description"]
    presentation_examples = TEMPLATES["presentation_examples"]
    presentation_template = TEMPLATES["presentation_template"]
    presentation_instruction=TEMPLATES["presentation_instruction"]
    presentation_question=TEMPLATES["presentation_question"]
    conversation_template=TEMPLATES["conversation_template"]

    # CREATE STRING FOR DATA, POST HOC DESCRIPTION and FEATURES DESCRIPTION
    data_string = get_data(loaded_model, idx, features, lime_path, MC_LIME_bool, LIME_bool, CEM_bool, average_class)
    post_hoc_description = define_post_hoc_description(LIME_bool, MC_LIME_bool, CEM_bool)
    features_description = get_features_description(features)

    # GET THE PROMPTING STRATEGIES
    instructions_dataset=pd.read_csv("prompting_strategies.csv")
    instructions_dataset=instructions_dataset[['name', 'Communication Prompt']]
    instructions_dataset = instructions_dataset[instructions_dataset['name'] != 'hesslow_contrastive']
    instructions_dataset = instructions_dataset.rename(columns={'Communication Prompt': 'instruction'})

    # Create prompt object
    prompt_object=prompt_class(template, instructions_dataset)

    # Create context partial prompt (prompt including only non-instructional parts of the template)
    prompt_object.context_partial_prompt( 
        data_string,
        features_description, 
        goal_definition, 
        model_description, 
        post_hoc_description, 
        course_description,
    )

    # Creates the prompts for each prompting strategy and saves it into prompt_object.prompting_strategies
    prompt_object.create_prompts_list(instructions_dataset)

    # Here we select a random prompt strategy
    test_prompt=prompt_object.prompting_strategies["prompt"][4]

    prompt_object.add_explanation_template(presentation_template)
    presentation_prompt=prompt_object.explanation_prompt_generator(presentation_instruction, presentation_question)

    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    # DEFINE LANGCHAIN MODEL
    llm_class=chat_openai_model("gpt-4-turbo", 0.5)
    conversation=llm_class.define_chat_model(conversation_template)

    print("\n\n______________________________FIRST PROMPT______________________________\n\n")
    print(langchain_prompt_to_str(test_prompt))
    # START CONVERSATION
    result=conversation.predict(input=test_prompt)
    print("\n\n______________________________FIRST ANSWER______________________________\n\n")
    print(result)
    print("\n\n______________________________PRESENTATION PROMPT______________________________\n\n")
    print(langchain_prompt_to_str(presentation_prompt))
    result=conversation.predict(input=presentation_prompt)

    print("\n\n______________________________FINAL RESULT______________________________\n\n")
    print(result)



if __name__ == "__main__":
    # Disable all warnings
    warnings.filterwarnings("ignore")

    # Call your main function
    main()