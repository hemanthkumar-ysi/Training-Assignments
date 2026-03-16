from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchResults

mcp = FastMCP("web-search-server")

search = DuckDuckGoSearchResults()


@mcp.tool()
async def web_search(query: str) -> str:
    """
    Search the internet for information.
    """

    results = search.run(query)
    print("Search Results:",str(results))
    return str(results)


if __name__ == "__main__":
    mcp.run()