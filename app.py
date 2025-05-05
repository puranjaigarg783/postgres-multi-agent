# client/claude_cli.py
import logging
import asyncio
import dotenv
import os
import sys
import json
import urllib.parse
import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client
from tabulate import tabulate
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.anthropic import Anthropic

tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_to_sql_and_run",
            "description": (
                "Take a plain text query from the user, convert to sql and run it on the database"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": ["query"]
            },
        },
    },
]


llm = Anthropic(model="claude-3-opus-20240229", api_key=os.getenv("ANTHROPIC_API_KEY"))

async def fetch_database_hierarchy(session, conn_id):
    """Fetch complete database structure from the MCP server using the new resource."""
    try:
        # Get the complete database information 
        db_resource = f"pgmcp://{conn_id}/"
        db_response = await session.read_resource(db_resource)
        
        content = None
        if hasattr(db_response, 'content') and db_response.content:
            content = db_response.content
        elif hasattr(db_response, 'contents') and db_response.contents:
            content = db_response.contents
            
        if content:
            content_item = content[0]
            if hasattr(content_item, 'text'):
                return json.loads(content_item.text)
        
        return None
    except Exception as e:
        print(f"Error fetching database hierarchy: {e}")
        return None

def format_database_hierarchy(db_structure):
    """Format the database structure in a hierarchical console output."""
    if not db_structure or 'schemas' not in db_structure:
        return "No database structure available."
    
    output = "DATABASE HIERARCHY:\n\n"
    
    for schema in db_structure['schemas']:
        schema_name = schema['name']
        schema_desc = schema.get('description', '')
        
        # Add schema header
        output += f"SCHEMA: {schema_name}\n"
        
        # Add tables for this schema
        for i, table in enumerate(schema['tables']):
            table_name = table['name']
            table_desc = table.get('description', '')
            row_count = table.get('row_count', 0)
            row_count_text = f" ({row_count} rows)" if row_count is not None else ""
            
            # Determine if this is the last table in the schema
            is_last_table = i == len(schema['tables']) - 1
            table_prefix = '└── ' if is_last_table else '├── '
            
            # Add table line
            output += f"{table_prefix}TABLE: {table_name}{row_count_text}\n"
            
            # Add columns
            for j, column in enumerate(table['columns']):
                column_name = column['name']
                column_type = column['type']
                
                # Gather constraints for this column
                constraints = []
                
                if not column['nullable']:
                    constraints.append('NOT NULL')
                
                if 'PRIMARY KEY' in column.get('constraints', []):
                    constraints.append('PRIMARY KEY')
                
                if 'UNIQUE' in column.get('constraints', []):
                    constraints.append('UNIQUE')
                
                # Check if this column is part of a foreign key
                for fk in table.get('foreign_keys', []):
                    if column_name in fk.get('columns', []):
                        ref_schema = fk.get('referenced_schema', '')
                        ref_table = fk.get('referenced_table', '')
                        ref_cols = fk.get('referenced_columns', [])
                        ref_col = ref_cols[fk.get('columns', []).index(column_name)] if ref_cols and column_name in fk.get('columns', []) else ''
                        
                        constraints.append(f"FK → {ref_schema}.{ref_table}({ref_col})")
                
                # Format constraints text
                constraints_text = f", {', '.join(constraints)}" if constraints else ""
                
                # Determine if this is the last column in the table
                is_last_column = j == len(table['columns']) - 1
                
                # Determine the appropriate prefix based on the nested level
                if is_last_table:
                    column_prefix = '    └── ' if is_last_column else '    ├── '
                else:
                    column_prefix = '│   └── ' if is_last_column else '│   ├── '
                
                # Add column line
                output += f"{column_prefix}{column_name}: {column_type}{constraints_text}\n"
            
            # Add description if available
            if table_desc:
                description_prefix = '    ' if is_last_table else '│   '
                output += f"{description_prefix}Description: {table_desc}\n"
            
            # Add vertical spacing between tables (except for the last table)
            if not is_last_table:
                output += "│\n"
        
        # Add vertical spacing between schemas
        if schema != db_structure['schemas'][-1]:
            output += "\n"
    
    return output

def clean_sql_query(sql_query):
    """
    Clean a SQL query by properly handling escaped quotes and trailing backslashes.
    
    Args:
        sql_query (str): The SQL query to clean
        
    Returns:
        str: Cleaned SQL query
    """
    # Handle escaped quotes - need to do this character by character to avoid issues with trailing backslashes
    result = ""
    i = 0
    
    while i < len(sql_query):
        if sql_query[i] == '\\' and i + 1 < len(sql_query):
            # This is an escape sequence
            if sql_query[i+1] == '"':
                # Convert escaped quote to regular quote
                result += '"'
                i += 2  # Skip both the backslash and the quote
            elif sql_query[i+1] == '\\':
                # Handle escaped backslash
                result += '\\'
                i += 2  # Skip both backslashes
            else:
                # Some other escape sequence, keep it
                result += sql_query[i:i+2]
                i += 2
        else:
            # Regular character
            result += sql_query[i]
            i += 1
    
    # Remove any extraneous whitespace or newlines
    result = result.strip()
    
    return result

async def generate_sql_with_anthropic(user_query, schema_text, anthropic_api_key)
                
                # Extract SQL and explanation
                sql_query = response_data.get("sql", "")
                explanation = response_data.get("explanation", "")
                
                # Print the results
                if explanation:
                    print(f"\nExplanation:")
                    print(f"------------")
                    print(explanation)
                
                # Original query (as generated by Claude)
                print(f"\nGenerated SQL query:")
                print(f"------------------")
                print(sql_query)
                print(f"------------------\n")
                
                if not sql_query:
                    print("No SQL query was generated. Exiting.")
                    sys.exit(1)
                
                # Clean the SQL query before execution
                sql_query = clean_sql_query(sql_query)
                
                # Show the cleaned query
                print(f"Cleaned SQL query:")
                print(f"------------------")
                print(sql_query)
                print(f"------------------\n")
                
                # Execute the generated SQL query
                print("Executing SQL query...")
                try:
                    result = await session.call_tool(
                        "pg_query", 
                        {
                            "query": sql_query,
                            "conn_id": conn_id
                        }
                    )
                    
                    # Extract and format results
                    if hasattr(result, 'content') and result.content:
                        print("\nQuery Results:")
                        print("==============")
                        
                        # Handle the different possible structures for results
                        if isinstance(result.content, list):
                            # Extract multiple text items from content array
                            query_results = []
                            for item in result.content:
                                if hasattr(item, 'text') and item.text:
                                    try:
                                        # Parse each text item as JSON
                                        row = json.loads(item.text)
                                        query_results.append(row)
                                    except json.JSONDecodeError:
                                        print(f"Warning: Could not parse result: {item.text}")
                        else:
                            # Legacy format handling
                            content = result.content[0]
                            if hasattr(content, 'text'):
                                try:
                                    query_results = json.loads(content.text)
                                except json.JSONDecodeError:
                                    print(f"Error: Could not parse content: {content.text}")
                                    query_results = []
                            else:
                                query_results = []
                        
                        if query_results:
                            # Pretty print the results
                            if isinstance(query_results, list) and len(query_results) > 0:
                                # Use tabulate to format the table
                                table = tabulate(
                                    query_results, 
                                    headers="keys",
                                    tablefmt="pretty",  # Options: "plain", "simple", "github", "grid", "fancy_grid", "pipe", "orgtbl", "jira", etc.
                                    numalign="right",
                                    stralign="left"
                                )
                                print(table)
                                print(f"\nTotal rows: {len(query_results)}")
                            elif isinstance(query_results, dict):
                                # Handle single dictionary case
                                table = tabulate(
                                    [query_results],
                                    headers="keys",
                                    tablefmt="pretty",
                                    numalign="right",
                                    stralign="left"
                                )
                                print(table)
                                print("\nTotal rows: 1")
                            else:
                                print(json.dumps(query_results, indent=2))
                        else:
                            print("Query executed successfully but returned no results.")

                    else:
                        print("Query executed but returned no content.")
                except Exception as e:
                    print(f"Error executing SQL query: {type(e).__name__}: {e}")
                    print(f"Failed query was: {sql_query}")
                
                # Disconnect when done
                print("Disconnecting from database...")
                try:
                    await session.call_tool(
                        "disconnect", 
                        {
                            "conn_id": conn_id
                        }
                    )
                    print("Successfully disconnected.")
                except Exception as e:
                    print(f"Error during disconnect: {e}")
                
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

# Create the agent with the required fields.
agent = FunctionAgent(
    name="DB Explorer",
    description="A helpful agent that can  execute a given query",
    tools=[convert_to_sql_and_run],
    llm=llm,
    system_prompt="You are a helpful assistant that can fetch data and information from a read only database",
)

workflow = AgentWorkflow(agents=[agent])

async def main():
    #logger.debug("Starting main function in workflow.")
    print("at 1")
    response = await workflow.run(user_msg="How many records in raw_survey_data")
    #response = await workflow.run(user_msg="fetch no of records in the industry_metrics table")
    #logger.debug("Workflow response received: %s", response)
    print(str(response))


if __name__ == "__main__":
    asyncio.run(main())
