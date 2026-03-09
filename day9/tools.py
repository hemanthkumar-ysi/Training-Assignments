from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Useful for searching information on the internet.
    Use this tool when you need facts, explanations, or recent information about a topic.
    Input should be a search query.
    """
    return search.run(query)


@tool
def write_file(content: str) -> str:
    """
    Use this tool to write research results into a text file.
    Input should be the final summarized research content.
    """
    with open("research_output.txt", "w", encoding="utf-8") as f:
        f.write(content)

    return "Research saved to research_output.txt"