# %%
import pandas as pd
import os

import numpy as np 
from dotenv import load_dotenv, find_dotenv

from prompt_templates import TEMPLATES

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import ast
import altair as alt
from tqdm import tqdm
import argparse
import json
import time
from chat_replicate import chat_replicate_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--COURSE', type=str, required=True, help='Course name')
    parser.add_argument('--MODEL', type=str, required=True, help='Model name')  # Added MODEL argument

    return parser.parse_args()

# %%
def def_prompt_template(template):
    """
    This function creates a ChatPromptTemplate object from the given template.
    
    Returns:
        prompt_template (ChatPromptTemplate): The created ChatPromptTemplate object.
    """
    # Create a ChatPromptTemplate object from the template
    prompt_template = ChatPromptTemplate.from_template(template)
    
    return prompt_template

# %%
# Function to generate evaluation
def generate_evaluation(row, model, questions, conversation_template, evaluation_template):
        llm_class = chat_replicate_model(model, 0.5)
        conversation = llm_class.define_chat_model(conversation_template)
        
        strategy = row['strategy']
        question_list = str(questions.get(strategy, []))
        if not question_list:
            return None, None
        
        formatted_question = evaluation_template.format(question=question_list)
        
        messages = [
            HumanMessage(content=row["first_prompt"]),
            AIMessage(content=row["response1"]),
        ]
        
        conversation.memory.chat_memory.messages = messages
        
        result = conversation.predict(input=formatted_question)

        return result, formatted_question

# Function to create one-hot encoding for answers
def one_hot_encode_answers(row):
    try:
        # Convert the string representation of the list into an actual list
        answers = ast.literal_eval(row)
    except:
        # Handle lists without quotes around 'YES' and 'NO'
        answers = row.strip('[]').split(', ')
    
    # Initialize a list with None for 9 positions
    encoded_answers = [None] * 9
    
    # Fill in the one-hot encoding based on the answers
    for i, answer in enumerate(answers):
        if i < 9:
            encoded_answers[i] = 1 if answer == 'YES' else 0
            
    return encoded_answers

# %%
# Assuming the DataFrame is named 'results_df' and the column is 'evaluation_answer1'
def calculate_yes_percentage(row):
    try:
        # Convert the string representation of the list into an actual list
        answers = ast.literal_eval(row)
        # Calculate the percentage of 'YES' in the list
        yes_percentage = answers.count('YES') / len(answers) * 100
        return yes_percentage
    except:
        try:
            # Handle lists without quotes around 'YES' and 'NO'
            answers = row.strip('[]').split(', ')
            yes_percentage = answers.count('YES') / len(answers) * 100
            return yes_percentage
        except:
            return np.nan

# %%
def main():
    args = parse_arguments()
    print("COURSE:", args.COURSE)
    print("MODEL:", args.MODEL)

    course = args.COURSE
    model = args.MODEL

    valid_courses = ["villesafricaines_001", "geomatique_003", "dsp_001"]
    
    assert course in valid_courses, f"Invalid course name: {course}. Must be one of {valid_courses}"

    # %%
    answers = pd.read_csv(f'data/final_results/gpt4o_answers_{course}.csv')

    # %%
    # instances_annotation_DSP = [230, 6478, 123, 5192, 16920]
    # instances_annotation_GEO = [226, 291, 138, 234, 174]
    # instances_annotation_VILLES = [1090, 2560, 2608, 3170, 1562]

    # # Conditional filtering based on the course variable
    # if course == "dsp_001":
    #     answers = answers[answers['student'].isin(instances_annotation_DSP)]
    # elif course == "geomatique_003":
    #     answers = answers[answers['student'].isin(instances_annotation_GEO)]
    # elif course == "villesafricaines_001":
    #     answers = answers[answers['student'].isin(instances_annotation_VILLES)]

    # assert len(answers) == len(answers['student'].unique()) * 24, f"There are {len(answers)} rows but there should be {len(answers['student'].unique())*24} rows"

    # %%

    # Specify the file path
    file_path = 'data/decomposed_questions.json'

    # Read the JSON file
    with open(file_path, 'r') as file:
        questions = json.load(file)

    # %%

    conversation_template = TEMPLATES['conversation_template']

    # %%
    # DEFINE REPLICATE MODEL

    llm_class = chat_replicate_model(model, 0.5)
    conversation = llm_class.define_chat_model(conversation_template)

    # %%

    eval_template = """ 
    Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. 

    Your selection should be based on your judgment as well as the following rules: 

    - YES: Select YES if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a YES rating. As an illustration, consider a question that asks, “Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be ‘YES’. To qualify for a YES rating, the generated text must be entirely accurate and relevant to the question. 

    - NO: Opt for NO if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks, “Is the second sentence in the generated text a compound sentence?” and the generated text only has one sentence, it offers no relevant information to answer the question. Consequently, the answer should be NO.

    QUESTIONS:
    {question}

    FORMAT: 
    A list of YES/NO answers separated by commas in a list format. Example: [answer1, answer2, ...]
    """

    # %%
    create_question_string = str(questions["abnormal_conditions"])

    # %%
    evaluation_template = def_prompt_template(eval_template)

    # %%
    # Apply the function to each row and create the new DataFrame
    results = []
    count = 0

    # Use tqdm with total=len(answers) to set the total number of iterations
    for idx, row in tqdm(answers.iterrows(), total=len(answers), desc="Processing rows"):
        result, question = generate_evaluation(row, model, questions, conversation_template, evaluation_template)
        count += 1
        results.append({
            "student": row["student"],
            "explainer": row["explainer"],
            "strategy": row["strategy"],
            "first_prompt": row["first_prompt"],
            "response1": row["response1"],
            "evaluation_prompt" : question,
            "evaluation_answer1": result
        })

    results_df = pd.DataFrame(results)

    # %%
    evaluation_results = results_df

    # %%

    # Apply the function to create one-hot encodings
    evaluation_results['one_hot_encoded'] = evaluation_results['evaluation_answer1'].apply(one_hot_encode_answers)

    # Split the one-hot encoded results into separate columns
    for i in range(9):
        evaluation_results[f'question_{i}'] = evaluation_results['one_hot_encoded'].apply(lambda x: x[i])

    # Drop the temporary 'one_hot_encoded' column
    evaluation_results.drop(columns=['one_hot_encoded'], inplace=True)

    # Apply the function to the DataFrame
    evaluation_results['yes_percentage'] = evaluation_results['evaluation_answer1'].apply(calculate_yes_percentage)
    
    # %%
    # Save or display the resulting DataFrame
    results_df.to_csv(f'data/final_results/{model}_evaluation_{course}.csv', index=False)

if __name__ == "__main__":
    main()
