import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


async def get_tools():

    client = MultiServerMCPClient(
        {
            "web": {
                "command": sys.executable,
                "args": ["mcp_servers/web_search_server.py"],
                "transport": "stdio",
            },
            "vectordb": {
                "command": sys.executable,
                "args": ["mcp_servers/vectordb_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    return tools