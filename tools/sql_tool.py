from typing import List, Optional
from uuid import UUID
from pydantic.v1 import BaseModel, Field
from langchain_core.pydantic_v1 import Field

# from langchain_community.callbacks import Callback
from langchain.sql_database import SQLDatabase
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools import BaseTool
from langchain.sql_database import SQLDatabase
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools import BaseTool
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    BaseSQLDatabaseTool
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit



class QuerySQLLimitedDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database with a limit on the output."""
    
    name: str = "sql_db_query"
    description: str = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        
        result = self.db.run_no_throw(query)
        
        results_list = eval(result)        
        if len(results_list) > 100:
            results_list = results_list[:100]
        
        return str(results_list)
    
class SQLDatabaseToolkitLimited(SQLDatabaseToolkit):
    
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )

        # Note that we're using the limited version of the query tool
        query_sql_database_tool = QuerySQLLimitedDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]



def build_sql_tool(db_uri, description, name='query_sql_db_tool', llm=None) -> StructuredTool:
    '''Builds a tool that can run sql queries against a database.'''
    

    toolkit = SQLDatabaseToolkitLimited(
        llm=ChatOpenAI(model='gpt-4', temperature=0), 
        db=SQLDatabase.from_uri(db_uri)
    )
    
    sql_agent = create_sql_agent(
        llm=llm or ChatOpenAI(model='gpt-4', temperature=0), 
        toolkit=toolkit,
        agent_type="openai-tools", 
        verbose=True
    )
    
    class SqlAgentInput(BaseModel):
        sql_query: str = Field()
    
    def sql_agent_run_wrapper(*args, **kwargs):
        '''Runs a sql query against the spaceship titanic database and returns the results.'''
        result = sql_agent.invoke(*args, **kwargs)

        if isinstance(result, dict):
            return result['output']
        return str(result)
            
    sql_agent_tool = StructuredTool.from_function(
        func=sql_agent_run_wrapper,
        name=name, 
        description=description,
        verbose=True,
        return_direct=True,
        args_schema=SqlAgentInput
    )
    
    return sql_agent_tool
    



if __name__ == '__main__':
    
    db_uri = 'sqlite:///./spaceship_titanic.db'
    # Example usage
    tool = build_sql_tool(
        db_uri=db_uri, 
        name='query_sql_db_tool', 
        description='Runs a sql query against the spaceship titanic database and returns the results.'
    )
    
    result = tool.invoke('Is there a difference in the percentage of passengers who spent more than 1000 at the food court between survivors and non-survivors?')
    print(result)