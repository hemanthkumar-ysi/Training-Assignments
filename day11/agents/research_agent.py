async def research_agent(state, tools):
    query = state["question"]

    tools_dict = {tool.name: tool for tool in tools}

    web_tool = tools_dict.get("web_search")

    if web_tool is None:
        state["research"] = "Web search tool not available."
        return state

    result = await web_tool.ainvoke({"query": query})

    state["research"] = result
    return state