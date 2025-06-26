# Tiny MCP CLI

A tiny CLI REPL for MCP (Model Context Protocol)

## Usage

```bash
uv run tiny-mcp
```

## Features

- Interactive REPL for MCP
- Rich terminal UI

## Requirements

- Python >= 3.10

## Dependencies

- mcp >= 0.5.0
- rich >= 13.0.0
- click >= 8.0.0
- python-dotenv >= 1.0.0
- openai >= 1.0.0
- pydantic >= 2.0.0

## How to add a new tool?

1. Create a new Python file in `tiny_mcp_cli/tools/` directory (e.g., `my_tool.py`)

2. Implement your tool function with proper type hints and docstring:
   ```python
   def my_tool(param1: str, param2: bool = False) -> Dict[str, Any]:
       """
       MCP Tool: Description of what your tool does
       
       Args:
           param1: Description of parameter 1
           param2: Description of parameter 2 (default: False)
       
       Returns:
           Dict containing the result and metadata
       """
       try:
           # Your tool implementation here
           result = do_something(param1, param2)
           
           return {
               "success": True,
               "result": result,
               # Add other relevant data
           }
       except Exception as e:
           return {
               "error": f"Error in my_tool: {str(e)}",
               "success": False
           }
   ```

3. Add the MCP Tool metadata at the end of your file:
   ```python
   TOOL_INFO = {
       "name": "my_tool",
       "description": "Brief description of your tool's functionality",
       "inputSchema": {
           "type": "object",
           "properties": {
               "param1": {
                   "type": "string",
                   "description": "Description of parameter 1"
               },
               "param2": {
                   "type": "boolean", 
                   "description": "Description of parameter 2",
                   "default": False
               }
           },
           "required": ["param1"]  # List required parameters
       }
   }
   ```

4. The tool will be automatically discovered and available in the MCP CLI
