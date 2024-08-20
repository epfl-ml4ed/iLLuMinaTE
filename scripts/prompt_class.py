
from langchain.prompts import ChatPromptTemplate

class prompt_class():

    def __init__(self, template, prompting_strategies):
        """
        Initializes an instance of the Prompt_generator class.
        
        Args:
            template (str): The template to be used for generating prompts.
            prompting_strategies (dataframe): A dataframe of prompting strategies to be used.
        """
        self.prompting_strategies = prompting_strategies
        self.complete_context = ""
        self.template = self.def_prompt_template(template)
        self.explanation_template = ""
        self.presentation_complete_context = ""
    
    def def_prompt_template(self, template):
        """
        This function creates a ChatPromptTemplate object from the given template.
        
        Returns:
            prompt_template (ChatPromptTemplate): The created ChatPromptTemplate object.
        """
        # Create a ChatPromptTemplate object from the template
        prompt_template = ChatPromptTemplate.from_template(template)
        
        return prompt_template
    
    def context_partial_prompt(self, data_string, features_description, goal_definition, model_description, post_hoc_description, student_description):
        self.complete_context = self.template.partial(
            data_string=data_string,
            features_description=features_description,
            goal_definition=goal_definition,
            model_description=model_description,
            post_hoc_description=post_hoc_description,
            course_description=student_description,
        )

        return self.complete_context
    
    def add_prompting_strategies(self, prompting_strategies):
        self.prompting_strategies = prompting_strategies
        return self.prompting_strategies
    
    def prompt_generator(self, instructions):
        formatted_prompt = self.complete_context.format_messages(
            instructions=instructions
        )
        return formatted_prompt
    
    def presentation_context_partial_prompt(self, course_description):
        self.presentation_complete_context = self.explanation_template.partial(
            course_description=course_description,
        )

        return self.presentation_complete_context
    
    def create_prompts_list(self, prompting_strategies):
        """
        Creates a dataframe containing the prompts for each prompting strategy in the columns "Prompt_1" and "Prompt_2".

        Parameters: 
        - prompting_strategies: A dataframe containing the prompting strategies.
        
        Returns:
        - dataframe: A dataframe containing the prompts for each prompting strategy in the columns "Prompt_1" and "Prompt_2".
        """
        # Initialize lists to store the prompts
        self.prompting_strategies = prompting_strategies
        prompt_1_list = []
        prompt_2_list = []

        # Iterate over each row in the prompting strategies dataframe
        for _, row in self.prompting_strategies.iterrows():
            # Create the prompt for instruction_1
            prompt_1 = self.prompt_generator(row['instruction'])
            prompt_1_list.append(prompt_1)
            if row['presentation_instruction']:
                # Create the prompt for instruction_2
                prompt_2 = self.explanation_prompt_generator(row['presentation_instruction'])
                prompt_2_list.append(prompt_2)

        # Add the lists as new columns in the prompting strategies dataframe
        self.prompting_strategies['prompt'] = prompt_1_list
        self.prompting_strategies['presentation_prompt'] = prompt_2_list
        self.prompting_strategies = self.prompting_strategies.drop(columns=['presentation_instruction'])

        return self.prompting_strategies
    
    def add_explanation_template(self, explanation_template):
        self.explanation_template = self.def_prompt_template(explanation_template)
        return self.explanation_template
    
    def explanation_prompt_generator(self, presentation_instruction):
        formatted_prompt = self.presentation_complete_context.format_messages(
            presentation_instruction=presentation_instruction,
        )
        return formatted_prompt
    