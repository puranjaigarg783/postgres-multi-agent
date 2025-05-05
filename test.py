# test.py
import asyncio
import httpx
import json
import sys
from mcp import ClientSession
from mcp.client.sse import sse_client

async def run(connection_string: str | None):
    """Test the MCP server with an optional database connection string."""
    # Assuming your server is running on localhost:8000
    server_url = "http://localhost:8000/sse"  
    
    try:
        print(f"Connecting to MCP server at {server_url}...")
        if connection_string:
            # Clean and sanitize the connection string
            clean_connection = connection_string.strip()
            # Only show a small part of the connection string for security
            masked_conn_string = clean_connection[:10] + "..." if len(clean_connection) > 10 else clean_connection
            print(f"Using database connection: {masked_conn_string}")
        
        # Create the SSE client context manager
        async with sse_client(url=server_url) as streams:
            print("SSE streams established, creating session...")
            
            # Create and initialize the MCP ClientSession
            async with ClientSession(*streams) as session:
                print("Session created, initializing...")
                
                # Initialize the connection
                await session.initialize()
                print("Connection initialized!")
                
                # List available prompts
                prompts_response = await session.list_prompts()
                print(f"Available prompts: {prompts_response}")
                
                # List available tools
                tools_response = await session.list_tools()
                tools = tools_response.tools
                print(f"Available tools: {[tool.name for tool in tools]}")
                                
                # List available resources
                resources_response = await session.list_resources()
                print(f"Available resources: {resources_response}")

                # List available resource templates
                templates_response = await session.list_resource_templates()
                print(f"Available resource templates: {templates_response}")

                # Test with a connection if provided
                if connection_string:
                    # Check if required tools are available
                    has_connect = any(tool.name == 'connect' for tool in tools)
                    has_pg_query = any(tool.name == 'pg_query' for tool in tools)
                    
                    if not has_connect:
                        print("\nERROR: 'connect' tool is not available on the server")
                        return
                    
                    if not has_pg_query:
                        print("\nERROR: 'pg_query' tool is not available on the server")
                        return
                        
                    try:
                        # Use the cleaned connection string
                        clean_connection = connection_string.strip()
                        
                        # First, register the connection to get a conn_id
                        print("\nRegistering connection with 'connect' tool...")
                        connect_result = await session.call_tool(
                            "connect", 
                            {
                                "connection_string": clean_connection
                            }
                        )
                        
                        # Extract conn_id from the response
                        conn_id = None
                        if hasattr(connect_result, 'content') and connect_result.content:
                            content = connect_result.content[0]
                            if hasattr(content, 'text'):
                                try:
                                    result_data = json.loads(content.text)
                                    conn_id = result_data.get('conn_id')
                                    print(f"Successfully connected with connection ID: {conn_id}")
                                except json.JSONDecodeError:
                                    print(f"Error parsing connect result: {content.text[:100]}")
                        
                        if not conn_id:
                            print("Failed to get connection ID from connect tool")
                            return
                        
                        # Test pg_query using the conn_id
                        print("\nTesting 'pg_query' tool with connection ID...")
                        query_result = await session.call_tool(
                            "pg_query", 
                            {
                                "query": "SELECT version() AS version",
                                "conn_id": conn_id
                            }
                        )
                        
                        # Process the query result
                        if hasattr(query_result, 'content') and query_result.content:
                            content = query_result.content[0]
                            if hasattr(content, 'text'):
                                try:
                                    version_data = json.loads(content.text)
                                    if isinstance(version_data, list) and len(version_data) > 0:
                                        print(f"Query executed successfully: {version_data[0].get('version', 'Unknown')}")
                                    else:
                                        print(f"Query executed successfully: {version_data}")
                                except json.JSONDecodeError:
                                    print(f"Error parsing query result: {content.text[:100]}")
                            else:
                                print("Query executed but text content not available")
                        else:
                            print("Query executed but no content returned")
                        
                        # Test pg_explain if available
                        has_pg_explain = any(tool.name == 'pg_explain' for tool in tools)
                        if has_pg_explain:
                            print("\nTesting 'pg_explain' tool...")
                            explain_result = await session.call_tool(
                                "pg_explain", 
                                {
                                    "query": "SELECT version()",
                                    "conn_id": conn_id
                                }
                            )
                            
                            if hasattr(explain_result, 'content') and explain_result.content:
                                content = explain_result.content[0]
                                if hasattr(content, 'text'):
                                    try:
                                        explain_data = json.loads(content.text)
                                        print(f"EXPLAIN query executed successfully. Result contains {len(explain_data)} rows.")
                                        # Pretty print a snippet of the execution plan
                                        print(json.dumps(explain_data, indent=2)[:500] + "...")
                                    except json.JSONDecodeError:
                                        print(f"Error parsing EXPLAIN result: {content.text[:100]}")
                        
                        # Test resources with the conn_id
                        print("\nTesting schema resources with connection ID...")
                        schema_resource = f"pgmcp://{conn_id}/schemas"
                        schema_response = await session.read_resource(schema_resource)
                        
                        # Process schema response
                        response_content = None
                        if hasattr(schema_response, 'content') and schema_response.content:
                            response_content = schema_response.content
                        elif hasattr(schema_response, 'contents') and schema_response.contents:
                            response_content = schema_response.contents
                        
                        if response_content:
                            content_item = response_content[0]
                            if hasattr(content_item, 'text'):
                                try:
                                    schemas_data = json.loads(content_item.text)
                                    print(f"Successfully retrieved {len(schemas_data)} schemas")
                                    
                                    # Print first few schemas
                                    for i, schema in enumerate(schemas_data[:3]):
                                        schema_name = schema.get('schema_name')
                                        print(f"  - {schema_name}")
                                        if i >= 2 and len(schemas_data) > 3:
                                            print(f"  ... and {len(schemas_data) - 3} more")
                                            break
                                    
                                    # If we have schemas, test extensions resource
                                    if schemas_data and len(schemas_data) > 0:
                                        schema_name = schemas_data[0].get('schema_name')
                                        print(f"\nTesting extensions for schema '{schema_name}'...")
                                        extensions_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/extensions"
                                        
                                        try:
                                            extensions_response = await session.read_resource(extensions_resource)
                                            
                                            # Process extensions response
                                            ext_content = None
                                            if hasattr(extensions_response, 'content') and extensions_response.content:
                                                ext_content = extensions_response.content
                                            elif hasattr(extensions_response, 'contents') and extensions_response.contents:
                                                ext_content = extensions_response.contents
                                            
                                            if ext_content:
                                                content_item = ext_content[0]
                                                if hasattr(content_item, 'text'):
                                                    extensions_data = json.loads(content_item.text)
                                                    print(f"Successfully retrieved {len(extensions_data)} extensions")
                                                    
                                                    # Print extensions and check for context
                                                    for ext in extensions_data:
                                                        has_context = ext.get('context_available', False)
                                                        context_flag = " (has context)" if has_context else ""
                                                        print(f"  - {ext.get('name')} v{ext.get('version')}{context_flag}")
                                                        
                                                        # If extension has context, test getting it
                                                        if has_context:
                                                            ext_name = ext.get('name')
                                                            print(f"\nFetching context for extension '{ext_name}'...")
                                                            context_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/extensions/{ext_name}"
                                                            
                                                            try:
                                                                context_response = await session.read_resource(context_resource)
                                                                
                                                                ctx_content = None
                                                                if hasattr(context_response, 'content') and context_response.content:
                                                                    ctx_content = context_response.content
                                                                elif hasattr(context_response, 'contents') and context_response.contents:
                                                                    ctx_content = context_response.contents
                                                                
                                                                if ctx_content:
                                                                    content_item = ctx_content[0]
                                                                    if hasattr(content_item, 'text'):
                                                                        try:
                                                                            context_data = content_item.text
                                                                            if isinstance(context_data, str) and context_data.strip():
                                                                                print(f"Retrieved context information for {ext_name}")
                                                                                # Don't print the whole context, just confirm it exists
                                                                                yaml_data = json.loads(context_data)
                                                                                print(f"Context contains sections: {', '.join(yaml_data.keys())}")
                                                                            else:
                                                                                print(f"Empty context received for {ext_name}")
                                                                        except json.JSONDecodeError:
                                                                            # Might be YAML directly
                                                                            print(f"Retrieved non-JSON context for {ext_name}")
                                                            except Exception as e:
                                                                print(f"Error fetching extension context: {e}")
                                        except Exception as e:
                                            print(f"Error fetching extensions: {e}")
                                                
                                    # Find a schema with tables to test table resources
                                    for schema_idx, schema in enumerate(schemas_data[:3]):
                                        schema_name = schema.get('schema_name')
                                        
                                        print(f"\nTesting tables for schema '{schema_name}'...")
                                        tables_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/tables"
                                        tables_response = await session.read_resource(tables_resource)
                                        
                                        # Process tables response
                                        tables_content = None
                                        if hasattr(tables_response, 'content') and tables_response.content:
                                            tables_content = tables_response.content
                                        elif hasattr(tables_response, 'contents') and tables_response.contents:
                                            tables_content = tables_response.contents
                                        
                                        if tables_content:
                                            content_item = tables_content[0]
                                            if hasattr(content_item, 'text'):
                                                tables_data = json.loads(content_item.text)
                                                print(f"Found {len(tables_data)} tables in schema '{schema_name}'")
                                                
                                                if tables_data and len(tables_data) > 0:
                                                    # Print first few tables
                                                    for i, table in enumerate(tables_data[:3]):
                                                        table_name = table.get('table_name')
                                                        print(f"  - {table_name}")
                                                        if i >= 2 and len(tables_data) > 3:
                                                            print(f"  ... and {len(tables_data) - 3} more")
                                                            break
                                                    
                                                    # Test table details for first table
                                                    table_name = tables_data[0].get('table_name')
                                                    print(f"\nTesting columns for table '{schema_name}.{table_name}'...")
                                                    
                                                    columns_resource = f"pgmcp://{conn_id}/schemas/{schema_name}/tables/{table_name}/columns"
                                                    columns_response = await session.read_resource(columns_resource)
                                                    
                                                    # Process columns response
                                                    cols_content = None
                                                    if hasattr(columns_response, 'content') and columns_response.content:
                                                        cols_content = columns_response.content
                                                    elif hasattr(columns_response, 'contents') and columns_response.contents:
                                                        cols_content = columns_response.contents
                                                    
                                                    if cols_content:
                                                        content_item = cols_content[0]
                                                        if hasattr(content_item, 'text'):
                                                            columns_data = json.loads(content_item.text)
                                                            print(f"Found {len(columns_data)} columns in table '{table_name}'")
                                                            
                                                            # Print first few columns
                                                            for i, col in enumerate(columns_data[:3]):
                                                                col_name = col.get('column_name')
                                                                data_type = col.get('data_type')
                                                                print(f"  - {col_name} ({data_type})")
                                                                if i >= 2 and len(columns_data) > 3:
                                                                    print(f"  ... and {len(columns_data) - 3} more")
                                                                    break
                                                    
                                                    # Test disconnect tool if available
                                                    break  # Exit schema loop once we've found a table
                                except json.JSONDecodeError:
                                    print(f"Error parsing schemas: {content_item.text[:100]}")
                        
                        # Finally, test the disconnect tool if available
                        has_disconnect = any(tool.name == 'disconnect' for tool in tools)
                        if has_disconnect and conn_id:
                            print("\nTesting 'disconnect' tool...")
                            disconnect_result = await session.call_tool(
                                "disconnect", 
                                {
                                    "conn_id": conn_id
                                }
                            )
                            
                            if hasattr(disconnect_result, 'content') and disconnect_result.content:
                                content = disconnect_result.content[0]
                                if hasattr(content, 'text'):
                                    try:
                                        result_data = json.loads(content.text)
                                        success = result_data.get('success', False)
                                        if success:
                                            print(f"Successfully disconnected connection {conn_id}")
                                        else:
                                            error = result_data.get('error', 'Unknown error')
                                            print(f"Failed to disconnect: {error}")
                                    except json.JSONDecodeError:
                                        print(f"Error parsing disconnect result: {content.text[:100]}")
                            else:
                                print("Disconnect call completed but no result returned")
                        
                    except Exception as e:
                        print(f"Error during connection tests: {e}")
                else:
                    print("\nNo connection string provided, skipping database tests")

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e}")
        print(f"Status code: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    except httpx.ConnectError:
        print(f"Connection Error: Could not connect to server at {server_url}")
        print("Make sure the server is running and the URL is correct")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    # Get database connection string from command line argument
    connection_string = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run(connection_string))
