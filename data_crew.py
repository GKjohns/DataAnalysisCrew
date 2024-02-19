import os
import re
import sys
sys.path.append('..')
from crewai import Agent, Task, Crew, Process
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.retrievers import ArxivRetriever
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

from Bronco import bronco
from prompts import CrewGenPrompts

from tools.sql_tool import build_sql_tool

def review_config(config, keep_file=False) -> str:
    
    # write the config to a file
    with open('crew_config.py', 'w') as f:
        f.write(str(config))
    
    # open the file in vim
    os.system('vim crew_config.py')
    
    # read the reviewed file back into memory
    with open('crew_config.py', 'r') as f:
        reviewed_config = f.read()
    
    # delete the file
    if not keep_file:
        os.system('rm crew_config.py')
    
    return eval(reviewed_config)

def extract_python_code(text):
    # Regular expression pattern to find the first code block marked as Python code
    pattern = r'```python(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    
    # If a match is not found, return the whole text string
    if not match:
        return text
    
    code = match.group(1).strip()  # Use .strip() to remove leading/trailing whitespace
    try:
        # Here, eval is not safe to use directly without proper context and validation
        return eval(code)
    except Exception as e:
        return f"Error evaluating code: {e}"
    


def generate_crew_config(objective, tool_names):
    crew_generator = bronco.LLMFunction(
        prompt_template=CrewGenPrompts.generate_crew_config_prompt,
        model_name=bronco.GPT_4,
        parser=extract_python_code
     )
    
    return crew_generator.generate({
        'objective': objective,
        'tool_names': tool_names
    })

def generate_agent_config(name, objective, agent_tasks, tool_names):
    """
    Generates an agent configuration for a Senior Research Analyst role with specific tasks and tools.
    
    Parameters:
    - name (str): The name of the agent.
    - agent_tasks (list): A list of tasks that the agent is responsible for.
    - tool_names (list): A list of tools that the agent has access to.
    
    Returns:
    - dict: The generated agent configuration.
    """

    
    agent_config_generator = bronco.LLMFunction(
        prompt_template=CrewGenPrompts.gen_agent_config_prompt,
        model_name=bronco.GPT_4,
        parser=extract_python_code,
    )
        
    # Generating agent configuration using the extracted values
    agent_config = agent_config_generator.generate({
        'name': name,
        'objective': objective,
        'agent_tasks': agent_tasks,
        'tool_names': tool_names
    })
        
    agent_config.update({'name': name})
    
    print(agent_config['name'])
    
    return agent_config

def generate_task_config(task_description, objective, agent_dict):
    """
    Generates a task configuration for creating a report on houseplant trends in the US in 2023.
    
    Parameters:
    - task_description (str): A brief description of the task.
    - objective (str): A detailed objective of what the report should cover.
    - agent_dict (dict): A dictionary containing the agent data.
    
    Returns:
    - dict: The generated task configuration.
    """
    
    # Extracting agent and tool_names from the agent_dict parameter
    agent_name = agent_dict['name']
    agent_role = agent_dict['role']
    tool_names = agent_dict.get('tool_names', [])
    
    task_config_generator = bronco.LLMFunction(
        prompt_template=CrewGenPrompts.gen_task_config_prompt,
        model_name=bronco.GPT_4,
        parser=extract_python_code,
        success_func=lambda x: 'description' in x and 'agent' in x
    )
    
    # Generating task configuration using the extracted values
    task_config = task_config_generator.generate({
        'task_description': task_description,
        'agent_role': agent_role,
        'objective': objective,
        'tool_names': tool_names
    })
    
    task_config.update({'name': agent_name})
    
    return task_config

def create_full_config(objective, tools, review_intermediate=True, keep_final_config=False):
    '''
    Create a full config for a crew based on an objective and a list of tools.
    '''
    tool_names = [tool.name for tool in tools]
    
    print('Generating crew config...')
    crew_config = generate_crew_config(objective, tool_names)
    
    if review_intermediate:
        crew_config = review_config(crew_config)
        
    # Create a config for each agent in the config
    agents = []
    for agent_name in crew_config['agents']:
        print(f'Generating agent config for {agent_name}...')
        agent_tasks = [task['task'] for task in crew_config['tasks'] if task['agent'] == agent_name]
        agent_config = generate_agent_config(
            name=agent_name, 
            objective=objective,
            agent_tasks=agent_tasks, 
            tool_names=tool_names
        )
        agents.append(agent_config)
        
    if review_intermediate:
        agents = review_config(agents)
                    
    # Create a config for each task in the config
    tasks = []
    for task in crew_config['tasks']:
        print(f'Generating task config for {task["task"]}...')
        task_agent = [agent for agent in agents if agent['name'] == task['agent']][0]
        task_description = task['task']
        
        task_config = generate_task_config(
            task_description=task_description,
            objective=objective, 
            agent_dict=task_agent
        )
        
        # string agent names need to be replaced with pointers to the agent objects
        # occurs during crew initialization, to ensure that we have a serializable config
        task_config.update({'agent': task_agent['name']})
                
        tasks.append(task_config)
    
    if review_intermediate:
        tasks = review_config(tasks)
    # create the full config
    crew_config = {
        'agents': agents,
        'tasks': tasks
    }
    
    # Allow the user to review the fully formed config
    review_config(crew_config, keep_file=keep_final_config)
    
    return crew_config


def initialize_from_config(config, verbose=2):
    '''
    Initialize a Crew object from a configuration dictionary.
    '''
    
    # agent tools need to be a pointer to the object, not a string
    for agent in config['agents']:
        agent['tools'] = [tool for tool in tools if tool.name in agent['tools']]
    agent_objects = [Agent(**agent) for agent in config['agents']]
    
    # the task agent needs to pe a pointer to the object, not a string
    agent_string_to_object = {}
    for agent_str, agent_obj in zip(config['agents'], agent_objects):
        agent_string_to_object[agent_str['name']] = agent_obj
    for task in config['tasks']:
        task['agent'] = agent_string_to_object[task['agent']]
    task_objects = [Task(**task) for task in config['tasks']]
    
    crew = Crew(
        agents=agent_objects, 
        tasks=task_objects, 
        verbose=verbose
    )
    
    return crew

def initialize_crew_from_saved_config(config_file, verbose=2):
    with open(config_file, 'r') as f:
        config = f.read()
    
    return initialize_from_config(eval(config), verbose=verbose)



if __name__ == '__main__':
    objective = (
        'Create a report on the impact of the following factors on survival rates'
        ' (probability of being successfully transported) on the spaceship titanic.'
        '\n- vip status'
        '\n- shopping and dining spending'
        '\n- cabin class'
    )
    
    sql_agent_tool = build_sql_tool(
        db_uri='sqlite:///./spaceship_titanic.db', 
        name='query_sql_db_tool', 
        description='Runs a sql query against the spaceship titanic database and returns the results.'
    )
    tools = [sql_agent_tool, PythonREPLTool()]
    
    print('Objective: ', objective)
    print('Tools: ', [tool.name for tool in tools])
        
    crew_config = create_full_config(
        objective=objective,
        tools=tools, 
        review_intermediate=False,
        keep_final_config=True
    )
    
    crew = initialize_from_config(crew_config)
    
    crew.kickoff()