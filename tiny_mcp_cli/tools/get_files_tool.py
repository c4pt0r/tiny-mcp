#!/usr/bin/env python3
"""
MCP Tool wrapper for get_files functionality
"""

import os
from pathlib import Path
from typing import Dict, Any, List


def get_files_tool(directory: str = '.', show_hidden: bool = False) -> Dict[str, Any]:
    """
    MCP Tool: Get files in the specified directory
    
    Args:
        directory: Directory to list (default: current directory)
        show_hidden: Show hidden files (default: False)
    
    Returns:
        Dict containing files, directories and metadata
    """
    path = Path(directory)
    
    if not path.exists():
        return {
            "error": f"Directory '{directory}' does not exist",
            "success": False
        }
    
    if not path.is_dir():
        return {
            "error": f"'{directory}' is not a directory",
            "success": False
        }
    
    files = []
    dirs = []
    
    try:
        for item in path.iterdir():
            if not show_hidden and item.name.startswith('.'):
                continue
            
            if item.is_file():
                files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "type": "file"
                })
            elif item.is_dir():
                dirs.append({
                    "name": item.name,
                    "type": "directory"
                })
        
        # Sort directories and files separately
        dirs.sort(key=lambda x: x["name"])
        files.sort(key=lambda x: x["name"])
        
        return {
            "success": True,
            "directory": str(path.absolute()),
            "directories": dirs,
            "files": files,
            "total_directories": len(dirs),
            "total_files": len(files)
        }
        
    except PermissionError:
        return {
            "error": f"Permission denied accessing '{directory}'",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Error accessing directory: {str(e)}",
            "success": False
        }


# MCP Tool metadata
TOOL_INFO = {
    "name": "get_files",
    "description": "List files and directories in a specified path with detailed information",
    "inputSchema": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory to list (default: current directory)",
                "default": "."
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Show hidden files and directories",
                "default": False
            }
        }
    }
}