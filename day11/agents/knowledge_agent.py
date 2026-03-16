async def knowledge_agent(state, tools):

    query = state.get("topic") or state["question"]

    tools_dict = {tool.name: tool for tool in tools}

    tool = tools_dict.get("search_knowledge")

    if tool is None:
        state["knowledge"] = "Knowledge base tool not available."
        return state

    result = await tool.ainvoke({"query": query})

    state["knowledge"] = result

    return state