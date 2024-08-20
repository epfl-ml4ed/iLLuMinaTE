# %%
import pandas as pd
from tqdm import tqdm

answers=pd.read_csv("data/final_results/llm_for_xai_results.csv")
answers

# %%
test=answers.loc[:2].copy()

# %%
import textstat
import language_tool_python

# Initialize the language tool for grammar and spell check
tool = language_tool_python.LanguageTool('en-US')

# Define a function to calculate readability scores and grammar issues
def calculate_text_stats(text):
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    matches = tool.check(text)
    grammar_issues = len(matches)
    return flesch_kincaid, gunning_fog, smog_index, grammar_issues




# %%
# File path to the existing CSV file
output_file_path = "data/final_results/results_metrics_no_CoT.csv"

output_df=pd.read_csv("data/final_results/llm_for_xai_results.csv")

# %%
# Step 2: Define the strings to be removed
strings_to_keep = [
    "[YES, YES, YES, YES, YES]",
    "[YES, YES, YES, NO, YES]",
    "[YES, YES, YES, YES, NO]",
    "[YES, YES, YES, NO, NO]"
]

# Step 2: Filter the DataFrame
# For rows where 'strategy' is 'chain_of_thought', only keep those with 'evaluation_answer1' in strings_to_keep
output_df = output_df[
    (output_df['strategy'] != "chain_of_thought") |
    (output_df['strategy'] == "chain_of_thought") & (output_df['model']=="gpt-4o")
]

# %%
output_df.to_csv(output_file_path, index=False)

# Ensure that the columns exist; if not, create them with NaN values
for column in ['Flesch_Kincaid', 'Gunning_Fog', 'SMOG_Index', 'Grammar_Issues']:
    if column not in output_df.columns:
        output_df[column] = pd.NA  # Initialize the column with NaN values

# %%
# Apply tqdm to the DataFrame's 'apply' method
tqdm.pandas(desc="Calculating text stats")

# Apply the function to each row with progress tracking
for index, row in tqdm(output_df.iterrows(), total=len(output_df), desc="Processing rows"):

    if pd.notna(row['Flesch_Kincaid']) and pd.notna(row['Gunning_Fog']) and pd.notna(row['SMOG_Index']) and pd.notna(row['Grammar_Issues']):
        continue  # Skip this row if all values are not NaN

    # Calculate the text stats for the current row
    flesch_kincaid, gunning_fog, smog_index, grammar_issues = calculate_text_stats(row['response1'])
    
    # Update the existing DataFrame with the new columns
    output_df.loc[index, 'Flesch_Kincaid'] = flesch_kincaid
    output_df.loc[index, 'Gunning_Fog'] = gunning_fog
    output_df.loc[index, 'SMOG_Index'] = smog_index
    output_df.loc[index, 'Grammar_Issues'] = grammar_issues
    
    # Save the updated DataFrame back to the file after each update
    output_df.to_csv(output_file_path, index=False)

print("All rows processed and saved.")

# %%
# # Calculate the average for each explainer
# explainer_averages = answers.groupby('explainer')[['Flesch_Kincaid', 'Gunning_Fog', 'SMOG_Index']].mean().reset_index()

# # Calculate the average for each strategy
# strategy_averages = answers.groupby('strategy')[['Flesch_Kincaid', 'Gunning_Fog', 'SMOG_Index']].mean().reset_index()

# course_averages = answers.groupby('course')[['Flesch_Kincaid', 'Gunning_Fog', 'SMOG_Index']].mean().reset_index()

# model_averages = answers.groupby('model')[['Flesch_Kincaid', 'Gunning_Fog', 'SMOG_Index']].mean().reset_index()


# %%
# latex_table=explainer_averages.to_latex(index=False)
# latex_table

# %%
# strategy_averages


