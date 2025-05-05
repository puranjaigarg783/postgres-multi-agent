#!/usr/bin/env python3
# app2.py - Database query tool using MCP and LLM
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                    "query": {
                        "type": "string",
                        "description": "The natural language query to convert to SQL and execute"
                    }
                },
                "required": ["query"]
            },
        },
    },
]


# Initialize the Anthropic LLM with environment variables (safely)
def get_anthropic_llm():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return None
    return Anthropic(model="claude-3-7-sonnet-20250219", api_key=api_key)

# Initialize after environment variables are loaded
llm = None

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

async def generate_sql_with_anthropic(user_query, schema_text, anthropic_api_key):
    """Generate SQL using Claude with response template prefilling."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    system_prompt = f"""You are an expert PostgreSQL developer who will translate a natural language query into a SQL query.

You must provide your response in JSON format with two required fields:
1. "explanation": A brief explanation of your approach to the query
2. "sql": The valid, executable PostgreSQL SQL query

Here is the database schema you will use:
{schema_text}
"""
    
    try:
        # Use response template prefilling to force Claude to produce JSON
        # This works by adding an assistant message that starts with the JSON structure
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": '{"explanation": "'}
            ]
        )
        
        # Extract the result
        result_text = response.content[0].text
        
        # Since we prefilled with '{"explanation": "', we need to ensure the JSON is complete
        # First check if the result already contains both fields
        if '"sql":' in result_text:
            # The response likely contains both fields, try to parse it as is
            try:
                # Make sure JSON is properly closed with a final brace if needed
                if not result_text.strip().endswith('}'):
                    result_text += '}'
                
                result_json = json.loads(result_text)
                
                # If parsing succeeded and has both required fields, return it
                if "explanation" in result_json and "sql" in result_json:
                    return result_json
            except json.JSONDecodeError:
                # If parsing failed, we'll continue with more clean-up attempts
                pass
            
        # If we're here, the JSON wasn't complete. Let's try to fix it.
        # Make sure there's a closing quote for explanation
        if '"sql":' not in result_text:
            result_text += '", "sql": ""}'
            
        # Now try to parse the fixed JSON
        try:
            result_json = json.loads(result_text)
            return result_json
        except json.JSONDecodeError:
            # If all attempts failed, extract what we can using string manipulation
            explanation = result_text.split('"sql":', 1)[0].strip()
            if explanation.endswith(','):
                explanation = explanation[:-1]
            if not explanation.endswith('"'):
                explanation += '"'
                
            # Try to extract SQL
            sql = ""
            if '"sql":' in result_text:
                sql_part = result_text.split('"sql":', 1)[1].strip()
                if sql_part.startswith('"'):
                    sql = sql_part.split('"', 2)[1]
                else:
                    # Handle the case where sql value isn't properly quoted
                    sql = sql_part.split('}', 1)[0].strip()
                    if sql.endswith('"'):
                        sql = sql[:-1]
            
            return {
                "explanation": explanation.replace('{"explanation": "', ''),
                "sql": sql
            }
            
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "explanation": f"Error: {str(e)}",
            "sql": ""
        }

async def convert_to_sql_and_run(query):
    """
    Convert a natural language query to SQL and execute it against the database.
    
    Args:
        query (str): Natural language query from the user
        
    Returns:
        str: Formatted results of the SQL query execution
    """
    print(f"Converting natural language query to SQL: '{query}'")
    
    # Load environment variables
    dotenv.load_dotenv()
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    print(f"API Key starts with: {anthropic_api_key[:10]}... (length: {len(anthropic_api_key)})")
    db_url = os.getenv('DATABASE_URL')
    pg_mcp_url = os.getenv('PG_MCP_URL', 'http://localhost:8000/sse')
    
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is not set.")
        sys.exit(1)
    
    if not anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)
    
    ## Check command line arguments
    #if len(sys.argv) < 2:
    #    print("Usage: python cli.py 'your natural language query'")
    #    sys.exit(1)
    
    user_query = query
    print(f"Processing query: {user_query}")
    
    # First, connect to MCP server to get schema information
    try:
        print(f"Connecting to MCP server at {pg_mcp_url} to fetch schema information...")
        
        # Create the SSE client context manager
        async with sse_client(url=pg_mcp_url) as streams:
            print("SSE streams established, creating session...")
            
            # Create and initialize the MCP ClientSession
            async with ClientSession(*streams) as session:
                print("Session created, initializing...")
                
                # Initialize the connection
                await session.initialize()
                print("Connection initialized!")
                
                # Use the connect tool to register the connection
                print("Registering connection with server...")
                try:
                    connect_result = await session.call_tool(
                        "connect", 
                        {
                            "connection_string": db_url
                        }
                    )
                    
                    # Extract connection ID
                    if hasattr(connect_result, 'content') and connect_result.content:
                        content = connect_result.content[0]
                        if hasattr(content, 'text'):
                            result_data = json.loads(content.text)
                            conn_id = result_data.get('conn_id')
                            print(f"Connection registered with ID: {conn_id}")
                        else:
                            print("Error: Connection response missing text content")
                            sys.exit(1)
                    else:
                        print("Error: Connection response missing content")
                        sys.exit(1)
                except Exception as e:
                    print(f"Error registering connection: {e}")
                    sys.exit(1)
                
                # Fetch database hierarchy
                print("Fetching database hierarchy information...")
                db_hierarchy = await fetch_database_hierarchy(session, conn_id)
                
                # Display the database hierarchy
                print("\nDatabase Structure:")
                print("==================\n")
                hierarchy_text = format_database_hierarchy(db_hierarchy)
                print(hierarchy_text)
                print("\n==================\n")
                
                # Use this hierarchy for Claude's prompt
                schema_text = hierarchy_text
                
                print(f"Retrieved database structure information.")
                
                # Generate SQL using Claude with schema context
                print("Generating SQL query with Claude...")
                response_data = await generate_sql_with_anthropic(user_query, schema_text, anthropic_api_key)
                
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
                    result_output = "\nQuery Results:\n==============\n"
                    
                    if hasattr(result, 'content') and result.content:
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
                                        result_output += f"Warning: Could not parse result: {item.text}\n"
                        else:
                            # Legacy format handling
                            content = result.content[0]
                            if hasattr(content, 'text'):
                                try:
                                    query_results = json.loads(content.text)
                                except json.JSONDecodeError:
                                    result_output += f"Error: Could not parse content: {content.text}\n"
                                    query_results = []
                            else:
                                query_results = []
                        
                        if query_results:
                            # Format the results as a table
                            if isinstance(query_results, list) and len(query_results) > 0:
                                # Use tabulate to format the table
                                table = tabulate(
                                    query_results,
                                    headers="keys",
                                    tablefmt="pretty",  # Options: "plain", "simple", "github", "grid", "fancy_grid", "pipe", "orgtbl", "jira", etc.
                                    numalign="right",
                                    stralign="left"
                                )
                                result_output += table + f"\n\nTotal rows: {len(query_results)}\n"
                            elif isinstance(query_results, dict):
                                # Handle single dictionary case
                                table = tabulate(
                                    [query_results],
                                    headers="keys",
                                    tablefmt="pretty",
                                    numalign="right",
                                    stralign="left"
                                )
                                result_output += table + "\n\nTotal rows: 1\n"
                            else:
                                result_output += json.dumps(query_results, indent=2) + "\n"
                        else:
                            result_output += "Query executed successfully but returned no results.\n"
                    else:
                        result_output += "Query executed but returned no content.\n"
                    
                    print(result_output)
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
                
                # Return the final result string
                return result_output
                
    except Exception as e:
        error_message = f"Error: {type(e).__name__}: {e}"
        print(error_message)
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return f"{error_message}\n\n{traceback_str}"

# Agent and workflow will be initialized in main()

async def main():
    # Load environment variables
    dotenv.load_dotenv()
    logger.info("Environment variables loaded")
    
    # Check for required environment variables
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        logger.error("DATABASE_URL environment variable is not set")
        print("ERROR: DATABASE_URL environment variable is not set.")
        sys.exit(1)
    
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is not set")
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)
    
    # Initialize the LLM
    llm = get_anthropic_llm()
    if not llm:
        logger.error("Failed to initialize Anthropic LLM")
        sys.exit(1)
    
    # Create the agent with the required fields
    agent = FunctionAgent(
        name="DB Explorer",
        description="A helpful agent that can execute a given query",
        tools=[convert_to_sql_and_run],
        llm=llm,
        system_prompt="You are a helpful assistant that can fetch data and information from a read only database",
    )
    
    # Initialize the workflow
    workflow = AgentWorkflow(agents=[agent])
    
    # Get query from command line argument or use default
    default_query = "Which tables exist in the database"
    user_query = default_query
    
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    
    logger.info(f"Starting database query workflow with query: '{user_query}'")
    print(f"Starting database query workflow...")
    print(f"Query: '{user_query}'")
    
    try:
        # Run the workflow with the query
        response = await workflow.run(user_query)
        print("\nWorkflow Response:")
        print("==================")
        print(str(response))
        return response
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}")
        print(f"ERROR: {str(e)}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    asyncio.run(main())
