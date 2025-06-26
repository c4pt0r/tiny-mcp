#!/usr/bin/env python3
"""
Tiny MCP CLI with rich UI and tool calling.
"""

import asyncio
import json
import os
import sys
import importlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.status import Status
from rich.traceback import install
from rich.theme import Theme
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import traceback

# Install rich traceback handler for better error display
install(show_locals=True)

# Suppress httpx logging to avoid noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Custom theme for better visual experience
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "user": "bold white",
    "assistant": "bold green",
    "tool": "bold magenta",
    "system": "dim white",
})

console = Console(theme=THEME)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None


class TinyMCPCLI:
    """Tiny MCP CLI with enhanced user experience"""
    
    def __init__(self):
        self.console = console
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, Any] = {}
        self.openai_client: Optional[OpenAI] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.load_local_tools()
    
    def load_local_tools(self):
        """Load tools from the local tools directory"""
        tools_dir = Path(__file__).parent / "tools"
        
        if not tools_dir.exists():
            return
        
        # Import all Python files in the tools directory
        for tool_file in tools_dir.glob("*_tool.py"):
            try:
                module_name = f"tiny_mcp_cli.tools.{tool_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for TOOL_INFO and corresponding function
                if hasattr(module, 'TOOL_INFO'):
                    tool_info = module.TOOL_INFO
                    tool_name = tool_info["name"]
                    
                    # Find the tool function (should match the tool name + "_tool")
                    tool_func_name = f"{tool_name}_tool"
                    if hasattr(module, tool_func_name):
                        tool_func = getattr(module, tool_func_name)
                        
                        self.tools[f"local.{tool_name}"] = {
                            "server": "local",
                            "name": tool_name,
                            "description": tool_info["description"],
                            "schema": tool_info["inputSchema"],
                            "function": tool_func
                        }
                        
                        self.console.print(f"✓ Loaded local tool: {tool_name}", style="success")
                    
            except Exception as e:
                self.console.print(f"✗ Failed to load tool {tool_file.name}: {e}", style="error")
        
    def load_config(self) -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations"""
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key == "":
            self.console.print("[error]OPENAI_API_KEY is not set[/error]")
            sys.exit(1)
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            
        # TODO: add preload MCP servers here
        servers = {}
        
        return servers
    
    async def connect_servers(self, server_configs: Dict[str, MCPServerConfig]):
        """Connect to MCP servers"""
        with Status("[info]Connecting to MCP servers...", console=self.console):
            for name, config in server_configs.items():
                try:
                    server_params = StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env or {}
                    )
                    
                    session = await stdio_client(server_params)
                    await session.initialize()
                    
                    self.sessions[name] = session
                    
                    # Load tools from this server
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        self.tools[f"{name}.{tool.name}"] = {
                            "server": name,
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema
                        }
                    
                    self.console.print(f"✓ Connected to {name}", style="success")
                    
                except Exception as e:
                    self.console.print(f"✗ Failed to connect to {name}: {e}", style="error")
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
```text
  _____ _            __  __  ___ ___ 
 |_   _(_)_ _ _  _  |  \\/  |/ __| _ \\
   | | | | ' \\ || | | |\\/| | (__|  _/
   |_| |_|_||_\\_, | |_|  |_|\\___|_|  
              |__/                   
```

TinyMCP, a tiny command-line interface for MCP (Model Context Protocol) with AI tool calling.

**Available commands:**
- `/help` - Show this help message
- `/tools` - List available tools, see `./tools` directory for more details
- `/test <tool_name> <json_args>` - Test a tool directly
- `/clear` - Clear conversation history
- `/quit` - Exit the application

Type your message to start chatting! The AI can automatically call tools when needed.
        """
        
        self.console.print(Panel(
            Markdown(welcome_text, code_theme="rrt"),
            title="[bold white]Welcome[/bold white]",
            border_style="white"
        ))
    
    def display_tools(self):
        """Display available tools in a table"""
        if not self.tools:
            self.console.print("No tools available", style="warning")
            return
            
        table = Table(title="Available Tools")
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Server", style="magenta")
        table.add_column("Description", style="white")
        
        for tool_id, tool_info in self.tools.items():
            table.add_row(
                tool_info["name"],
                tool_info["server"],
                tool_info["description"]
            )
        
        self.console.print(table)
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool on the specified server or locally"""
        # Handle local tools
        if server_name == "local":
            tool_key = f"local.{tool_name}"
            if tool_key in self.tools and "function" in self.tools[tool_key]:
                try:
                    result = self.tools[tool_key]["function"](**arguments)
                    if isinstance(result, dict):
                        return json.dumps(result, indent=2)
                    else:
                        return str(result)
                except Exception as e:
                    return f"Error executing local tool: {e}"
            else:
                return f"Local tool {tool_name} not found"
        
        # Handle remote MCP server tools
        if server_name not in self.sessions:
            return f"Server {server_name} not available"
            
        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, arguments)
            
            if result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                elif isinstance(content, str):
                    return content
                else:
                    return json.dumps(content.__dict__ if hasattr(content, '__dict__') else content, indent=2)
            else:
                return "Tool executed successfully (no output)"
                
        except Exception as e:
            return f"Error executing tool: {e}"
    
    def format_message(self, role: str, content: str) -> Panel:
        """Format a message with appropriate styling"""
        if role == "user":
            return Panel(content, title="[user]You[/user]", border_style="white")
        elif role == "assistant":
            return Panel(
                Markdown(content, code_theme="rrt") if content.strip().startswith("#") or "\n" in content else content,
                title="[assistant]Assistant[/assistant]",
                border_style="green"
            )
        elif role == "tool":
            return Panel(
                Syntax(content, "json", theme="rrt") if content.strip().startswith("{") else content,
                title="[tool]Tool Output[/tool]",
                border_style="magenta"
            )
        else:
            return Panel(content, title="[system]System[/system]", border_style="dim")
    
    async def get_ai_response(self, message: str, call_depth: int = 0, max_depth: int = 5) -> str:
        """Get response from AI model with tool calling support and chain calling"""
        if not self.openai_client:
            return "No AI client configured. Please set OPENAI_API_KEY environment variable."
        
        # Prevent infinite recursion
        if call_depth >= max_depth:
            return f"❌ Maximum tool call depth ({max_depth}) reached. Stopping to prevent infinite recursion."
        
        try:
            # Build system prompt with available tools
            system_prompt = """You are a helpful assistant with access to various tools. When you need to use a tool, respond with a JSON object in this exact format:

{
  "tool": "tool-name",
  "arguments": {
    "argument-name": "value"
  }
}

If you don't need to use a tool, respond naturally with text. But if you need to use a tool, only respond a valid JSON string in the exact format above, do not respond with any other text.

Available tools:
"""
            for tool_id, tool_info in self.tools.items():
                system_prompt += f"- {tool_info['name']}: {tool_info['description']}\n"
                if "schema" in tool_info:
                    system_prompt += f"  Schema: {json.dumps(tool_info['schema'], indent=2)}\n"
            
            system_prompt += "\nIf you don't need to use a tool, respond naturally with text."
            
            # Add conversation history
            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            messages.append({"role": "user", "content": message})
            
            # do not output log when calling openai
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
            )
            
            ai_response = response.choices[0].message.content
            
            # Check if the response is a tool call
            if await self.is_tool_call(ai_response):
                return await self.process_tool_call(ai_response, call_depth)
            else:
                return ai_response
                
        except Exception as e:
            return f"Error getting AI response: {e}"
    
    async def is_tool_call(self, response: str) -> bool:
        """Check if the AI response is a tool call"""
        try:
            data = json.loads(response.strip())
            return isinstance(data, dict) and "tool" in data and "arguments" in data
        except (json.JSONDecodeError, AttributeError):
            return False
    
    async def process_tool_call(self, response: str, call_depth: int = 0) -> str:
        """Process a tool call from the AI response with chain calling support"""
        try:
            tool_call = json.loads(response.strip())
            tool_name = tool_call["tool"]
            arguments = tool_call.get("arguments", {})
            
            # Find the tool
            tool_key = None
            server_name = None
            
            # Check local tools first
            if f"local.{tool_name}" in self.tools:
                tool_key = f"local.{tool_name}"
                server_name = "local"
            else:
                # Check server tools
                for tid, tool_info in self.tools.items():
                    if tool_info["name"] == tool_name:
                        tool_key = tid
                        server_name = tool_info["server"]
                        break
            
            if not tool_key:
                return f"❌ Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            
            # Display tool execution
            self.console.print(f"\n🔧 Executing tool: [tool]{tool_name}[/tool]")
            self.console.print(f"Arguments: [dim]{json.dumps(arguments, indent=2)}[/dim]")
            
            # Execute the tool
            result = await self.execute_tool(server_name, tool_name, arguments)
            
            # Display tool result
            self.console.print(self.format_message("tool", result))
            
            # Add tool call and result to conversation history
            self.conversation_history.append({
                "role": "assistant", 
                "content": f"🔧 Used tool '{tool_name}' with arguments: {json.dumps(arguments)}"
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": f"Tool result: {result}"
            })
            
            # Continue the conversation with the tool result to see if AI needs more tools or can answer
            follow_up_prompt = f"""Based on the tool result above, please continue. If you need to use another tool to complete the task, use it. If you have enough information to answer the original question, provide the final answer."""
            
            return await self.get_ai_response(follow_up_prompt, call_depth + 1)
            
        except Exception as e:
            return f"❌ Error processing tool call: {e}, {traceback.format_exc()}"
    
    async def get_ai_interpretation(self, prompt: str) -> str:
        """Get AI interpretation of tool results without triggering more tool calls"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide a clear, natural language summary of the tool results. Do not use any tools - just respond with text."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error interpreting results: {e}"
    
    async def run_repl(self):
        """Run the main REPL loop"""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = prompt("\n[You]: ", style=Style.from_dict({
                    'prompt': 'bold white',
                    '': 'white'
                }))
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    await self.handle_command(user_input)
                    continue
                
                # Display user message
                self.console.print(self.format_message("user", user_input))
                
                # Get AI response
                with Status("[info]Thinking...", console=self.console):
                    response = await self.get_ai_response(user_input)
                
                # Display AI response
                self.console.print(self.format_message("assistant", response))
                
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                self.console.print("\n[warning]Interrupted by user[/warning]")
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[error]Error: {e}[/error]")
    
    async def handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            self.display_welcome()
        elif cmd == "/tools":
            self.display_tools()
        elif cmd.startswith("/test"):
            await self.test_tool_call(command)
        elif cmd == "/clear":
            self.conversation_history.clear()
            self.console.clear()
            self.console.print("[success]Conversation history cleared[/success]")
        elif cmd == "/quit":
            self.console.print("[info]Goodbye![/info]")
            sys.exit(0)
        else:
            self.console.print(f"[warning]Unknown command: {command}[/warning]")
    
    async def test_tool_call(self, command: str):
        """Test a tool call manually"""
        # Example: /test get_files {"directory": ".", "show_hidden": false}
        parts = command.split(None, 2)
        if len(parts) < 3:
            self.console.print("[warning]Usage: /test <tool_name> <json_arguments>[/warning]")
            return
        
        tool_name = parts[1]
        try:
            arguments = json.loads(parts[2])
        except json.JSONDecodeError:
            self.console.print("[error]Invalid JSON arguments[/error]")
            return
        
        # Find the tool
        tool_key = None
        server_name = None
        
        if f"local.{tool_name}" in self.tools:
            tool_key = f"local.{tool_name}"
            server_name = "local"
        else:
            for tid, tool_info in self.tools.items():
                if tool_info["name"] == tool_name:
                    tool_key = tid
                    server_name = tool_info["server"]
                    break
        
        if not tool_key:
            self.console.print(f"[error]Tool '{tool_name}' not found[/error]")
            return
        
        self.console.print(f"🧪 Testing tool: [tool]{tool_name}[/tool]")
        result = await self.execute_tool(server_name, tool_name, arguments)
        self.console.print(self.format_message("tool", result))
    
    async def cleanup(self):
        """Clean up resources"""
        for session in self.sessions.values():
            try:
                await session.close()
            except:
                pass

@click.command()
@click.option("--config", help="Path to configuration file")
def main(config: Optional[str] = None):
    """Tiny MCP CLI - A tiny command-line interface for MCP"""
    
    async def run():
        cli = TinyMCPCLI()
        try:
            # Load and connect to servers
            server_configs = cli.load_config()
            await cli.connect_servers(server_configs)
            # Run the REPL
            await cli.run_repl()
            
        except KeyboardInterrupt:
            console.print("\n[info]Shutting down...[/info]")
        finally:
            await cli.cleanup()
    # Run the async main function
    import nest_asyncio
    nest_asyncio.apply()
    
    # Now we can safely use asyncio.run even if there's a running loop
    asyncio.run(run())

if __name__ == "__main__":
    main()