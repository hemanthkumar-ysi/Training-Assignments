import asyncio
from tools.mcp_client import get_tools
from workflow.graph import create_graph


async def main():

    tools = await get_tools()

    graph = create_graph(tools)

    print("AI Assistant (type 'exit' to quit)")
    
    while True:
        question = input("\nAsk question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        state = {"question": question}

        result = await graph.ainvoke(state)

        print("\nFINAL ANSWER\n")
        print(result["final"])


if __name__ == "__main__":
    asyncio.run(main())