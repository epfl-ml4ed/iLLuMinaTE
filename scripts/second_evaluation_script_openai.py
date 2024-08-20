# %%
import pandas as pd
import os
import openai
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
from chat_open_ai import chat_openai_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--COURSE', type=str, required=True, help='Course name')
    parser.add_argument('--MODEL', type=str, required=True, help='Model name of the results to evaluate')
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
        llm_class = chat_openai_model(model, 0.5)
        conversation = llm_class.define_chat_model(conversation_template)
        
        strategy = row['strategy']
        question_list = str(questions.get(strategy, []))
        if not question_list:
            return None, None
        
        formatted_question = evaluation_template.format(question=question_list)
        
        messages = [
            HumanMessage(content=row["second_prompt"]),
            AIMessage(content=row["response2"]),
        ]
        
        conversation.memory.chat_memory.messages = messages
        
        result = conversation.predict(input=formatted_question)

        return result, formatted_question

# Function to create one-hot encoding for answers
def one_hot_encode_answers(row, max_num_questions=9):
    try:
        # Convert the string representation of the list into an actual list
        answers = ast.literal_eval(row)
    except:
        # Handle lists without quotes around 'YES' and 'NO'
        answers = row.strip('[]').split(', ')
    
    # Initialize a list with None for X positions
    encoded_answers = [None] * max_num_questions
    
    # Fill in the one-hot encoding based on the answers
    for i, answer in enumerate(answers):
        if i < max_num_questions:
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
    model_eval_name=args.MODEL
    model = "gpt-4o"

    print(f"MODEL NAME: {model_eval_name}")
    if "/" in model_eval_name:
        model_name_file_path = model_eval_name.split("/")[-1]
    else:
        model_name_file_path = model_eval_name

    file_path = f"data/{model_name_file_path}_answers_{course}_completed.csv"

    print(f"FILE PATH to save: {file_path}")

    print(f"model name to save: {model_name_file_path}\n\n\n")

    valid_courses = ["villesafricaines_001", "geomatique_003", "dsp_001"]
    
    assert course in valid_courses, f"Invalid course name: {course}. Must be one of {valid_courses}"

    # %%
    answers_path=f'data/final_results/{model_eval_name}_answers_{course}_completed.csv'
    answers = pd.read_csv(answers_path)

    print(f"\n\nANSWERS FILE: ", answers_path)

    # %%

    # Specify the file path
    file_path = 'data/second_step_decomposed_questions.json'

    # Read the JSON file
    with open(file_path, 'r') as file:
        questions = json.load(file)

    max_num_questions = max(len(questions) for questions in questions.values())
    print(f"Max number of questions: {max_num_questions}")


    # %%

    conversation_template = TEMPLATES['conversation_template']

    # %%
    # DEFINE OPENAI MODEL
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    llm_class = chat_openai_model(model, 0.5)
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

    columns = ['student', 'explainer', 'strategy', 'second_prompt', 'response2', 'evaluation_prompt', 'evaluation_answer2']

    # %%
    # Apply the function to each row and create the new DataFrame
    results_temp = pd.DataFrame(columns=columns)
    temp_file_path=f'data/final_results/second_evaluation_{model_name_file_path}_{course}_temp.csv'
    count = 0

    # Use tqdm with total=len(answers) to set the total number of iterations
    for idx, row in tqdm(answers.iterrows(), total=len(answers), desc="Processing rows"):
        #result, question = generate_evaluation(row, model, questions, conversation_template, evaluation_template)
        count += 1
        new_row = {
            "student": row["student"],
            "explainer": row["explainer"],
            "strategy": row["strategy"],
            "first_prompt": row["second_prompt"],
            "response2": row["response2"],
            "evaluation_prompt" : "question",
            "evaluation_answer2": "result"
        }

        results_temp = pd.concat([results_temp, pd.DataFrame([new_row])], ignore_index=True)

        # Save the results_temp DataFrame to the CSV file after each loop
        results_temp.to_csv(temp_file_path, index=False)

    # %%
    evaluation_results = results_temp

    # %%

    # Apply the function to create one-hot encodings
    evaluation_results['one_hot_encoded'] = evaluation_results['evaluation_answer2'].apply(one_hot_encode_answers, max_num_questions)

    # Split the one-hot encoded results into separate columns
    for i in range(max_num_questions):
        evaluation_results[f'question_{i}'] = evaluation_results['one_hot_encoded'].apply(lambda x: x[i])

    # Drop the temporary 'one_hot_encoded' column
    evaluation_results.drop(columns=['one_hot_encoded'], inplace=True)

    # Apply the function to the DataFrame
    evaluation_results['yes_percentage'] = evaluation_results['evaluation_answer2'].apply(calculate_yes_percentage)
    
    # %%
    # Save or display the resulting DataFrame
    evaluation_results.to_csv(f'data/final_results/second_evaluation_{model_name_file_path}_{course}_completed.csv', index=False)

if __name__ == "__main__":
    main()
