from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

class State(TypedDict):
    step: int

def process(state):
    step = state.get("step", 0)
    print("Current step:", step)
    return {"step": step + 1}

graph = StateGraph(State)

graph.add_node("process", process)
graph.set_entry_point("process")
graph.add_edge("process", END)

memory = MemorySaver()

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user1"}}

print("Run 1")
app.invoke({"step": 0}, config=config)

print("Run 2")
app.invoke({}, config=config)

print("Run 3")
app.invoke({}, config=config)