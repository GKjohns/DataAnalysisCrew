class DataQaPrompts:
    
    question_to_query_prompt = '''
    # Context
    Your job is to write a sql query that answers the following question:
    {question}

    Below is a list of columns and sample values. Your query should only use the data contained in the table. The table name is `{table_name}`.

    # Columns and sample values
    {table_sample}

    If the question is not a question or is answerable with the given columns, respond to the best of your ability.
    Do not use columns that aren't in the table.
    Ensure that the query runs and returns the correct output.

    # Your query:
    '''
    
    results_to_answer_prompt = '''
    # Task
    Based on the results of a SQL query, provide a brief summary of the key findings and explicitly answer the following question:
    {question}

    The query results from the table `{table_name}` are as follows:

    # Query Results Table
    {query_results_table}

    In 2-5 sentences, summarize the main insights from the query results and give a clear and direct answer to the original question.

    # Summary and Answer:
    '''
    
    

class CrewGenPrompts:
    generate_crew_config_prompt = '''
    Here's an example of configs for a simple agent and a simple task:
    ```python
    i{{
        'agents': [
            'content_writer',  # Writes engaging descriptions for plants and company info.
            'web_designer',  # Creates the layout and design for the landing page.
            'web_developer',  # Builds the website using HTML, CSS, and potentially JavaScript.
            'seo_specialist'  # Optimizes content for search engines to increase visibility.
        ],
        'tasks': [
             'tasks': [
            {{'task': 'write_plant_descriptions', 'agent': 'content_writer'}},
            {{'task': 'design_page_layout', 'agent': 'web_designer'}},
            {{'task': 'select_images', 'agent': 'web_designer'}},
            {{'task': 'write_about_us', 'agent': 'content_writer'}},
            {{'task': 'build_webpage', 'agent': 'web_developer'}},
            {{'task': 'implement_seo_practices', 'agent': 'seo_specialist'}},
            {{'task': 'setup_contact_form', 'agent': 'web_developer'}},
            {{'task': 'launch_page_review', 'agent': 'web_designer'}}
    ]
        ]
    }}
    ```
    
    # Task
    Create a list of agents and tasks that would complete the following obective: {objective} 
    The output should be a config in the form of a dictionary:
    ```
    {{'agents': ['agent1', 'agent1', ...], 'tasks': [...]}}
    ```

    Ensure that the agents and tasks are relevant to the objective and that the agents have the necessary skills to complete the tasks.
    Remember that each task must be delegated to an agent. Do not create a task that cannot be completed by any of the agents.
    Tasks and agent capabilities should be within the abilities that can be completed by a python coder with access to the internet and a powerful LLM AI.
    
    # Available tools
    Here's a list of tools the agents can use to complete their tasks
    Note that if a step involves analyzing data, there needs to be a task that involves acquiring the data:
    {tool_names}
    
    # Your crew config
    '''

    gen_agent_config_prompt = '''

    # Instructions
    The overall objective of the larger program is: {objective}
    Create a config for an agent with the name {name}, in the format below.
    The agent will have to complete the following tasks: {agent_tasks}
    The agent may make use of any of the following tools: {tool_names}
    
    ENSURE THAT YOUR CONFIG IS IN THE FORM OF A DICTIONARY WITH THE KEYS BELOW.
    Here's an example of a config for a simple agent:
    ```python
    {{
        'role': 'Unicorn Hunter',
        'goal': 'Discover and capture mythical unicorns for study and conservation',
        'backstory': \'''You are part of an ancient society dedicated to the preservation and study of unicorns. With a deep understanding of mythical creatures and their habitats, you embark on expeditions into enchanted forests. Your skills in tracking, magical lore, and non-lethal capture techniques are unparalleled. You work to ensure the survival of unicorns and the balance of their ecosystems, often collaborating with wizards and other mythical beings.\''',
        'verbose': True,
        'allow_delegation': False,
        'tools': ['enchanted_net', 'potion_brewing_kit', 'ancient_tome_of_lore']
    }}
    ```
    
    Do not duplicate tasks. If a task is already assigned to another agent, do not assign it to this agent.
    
    # Your agent config for {name}
    '''

    gen_task_config_prompt = '''
    # Instructions
    The overall objective of the larger program is: {objective}
    Create a config for a task with the following description: {task_description}
    The task should be delegated to the following agent: {agent_role}
    The agent will have access to the following tools: {tool_names}

    ENSURE THAT YOUR CONFIG IS IN THE FORM OF A PYTHON DICTIONARY with the keys 'description' and 'agent' only.
    Here's an example of a config for a simple task:
    ```python
    {{
    'description': \'''Conduct a comprehensive analysis of the latest advancements in AI in 2024.
    Identify key trends, breakthrough technologies, and potential industry impacts.
    Your final answer MUST be a full analysis report\''',
    'agent': 'researcher'
    }}
    ```
    
    # Your task config for {task_description}
    '''

    code_fixer_prompt = '''
    # Task
    Refine the formatting and fix any error in the given code snippet.
    Your primary goal is to ensure that the code will run successfully without any errors.
    Ensure brackets, quotes, and parentheses are balanced and properly nested.
    If the code is already well-formatted, return it unchanged.
    ONLY OUTPUT THE FIXED OR UNCHANGED CODE.

    # Language/File Type
    {language_or_file_type}

    # Input code snippet
    ```
    {code_snippet}
    ```
    
    # Output code snippet
    '''
    
    
if __name__ == '__main__':
    
    task_description = 'build_webpage'
    objective = (
        'Create a report on trends in houseplant in the US in 2023. '
        'The report should include a comprehensive analysis of the market, '
        'including the most popular plants, the most popular planters, '
        'and the most popular plant care products. '
        'The report should also include a section on the most popular houseplant influencers '
        'and a section on the most popular houseplant hashtags.'
    )
    tool_names = ['bing_search', 'code_pad']
    
    
    task_config_generator = bronco.LLMFunction(
        prompt_template=DataQaPrompts.gen_task_config_prompt
    )
    
    task_config = task_config_generator.generate({
        'task_description': task_description,
        'agent': 'web_developer',
        'objective': objective,
        'tool_names': tool_names
    })
    
    print(task_config)