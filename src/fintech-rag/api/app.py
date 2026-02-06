import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field
from typing import Protocol



load_dotenv()
app = FastAPI(title="Langchain Agent Demo")

# BEST PRACTICE: Use environment variable or config
# Set STATIC_DIR in your environment or .env file
STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))

# For development: make it work regardless of where you run from
if not STATIC_DIR.exists():
    # Fallback: search up from current file
    current = Path(__file__).resolve()
    for parent in current.parents:
        potential_static = parent / "static"
        if potential_static.exists():
            STATIC_DIR = potential_static
            break
    else:
        raise RuntimeError(
            f"Static directory not found. Please set STATIC_DIR environment variable."
        )

print(f"Using static directory: {STATIC_DIR.absolute()}")

# Mount static files for serving CSS, JS, images, etc.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# define the date/time tool
@tool
def get_current_datetime(query: str = "") -> str:
    """
    Returns the current data and time.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_sunset_time(city: str) -> str:
    """Get sunset time for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"The sunset time in {city} is 5:25pm!"

# Initialize Zhipu AI model


def get_agent():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("ZHIPUAI_API_KEY environment variable not set")
    llm = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.7)

    agent = create_agent(llm, tools=[get_current_datetime, get_weather, get_sunset_time])
    return agent
def _render_message_chunk(token: AIMessageChunk) -> str:
    if token.text:
        return token.text
    if token.tool_call_chunks:
        return str(token.tool_call_chunks)
    return ""

def _render_completed_message(message: AnyMessage) -> str:
    if isinstance(message, AIMessage):
        if message.tool_calls:
            return f"Tool Calls: {message.tool_calls}"
        return str(message.content)
    if isinstance(message, ToolMessage):
        return f"Tool Result: {message.content}"
    return str(message)

# define API, Request and Response
# TODO: pydantic?
class QueryRequest(BaseModel):
    query: str


class StepInfo(BaseModel):
    step: str
    message: str

class StreamModeInfo(BaseModel):
    stream_mode: str
    steps: list[StepInfo] = []   

class QueryResponse(BaseModel):
    response: str
    streamModes: list[StreamModeInfo] = []
    error: str | None = None


class AgentResponseHandler(Protocol):
    """Protocol for handling different types of agent responses"""

    def handle_response(self, agent, messages: list) -> tuple[str, list[StreamModeInfo]]:
        """Handle agent response and return final response and steps"""
        ...


class StreamResponseHandler:
    """Handler for streaming agent responses"""

    def handle_response(self, agent, messages: list) -> tuple[str, list[StreamModeInfo]]:
        # Group steps by stream mode
        modes_map: dict[str, list[StepInfo]] = {
            "messages": [],
            "updates": [],
            "custom": []
        }
        final_response = ""

        for stream_mode, chunk in agent.stream({"messages": messages}, stream_mode=["messages", "updates", "custom"]):
            if stream_mode == "messages":
                # messages mode: yields (token, metadata) tuples
                token, metadata = chunk
                if isinstance(token, AIMessageChunk):
                    content = _render_message_chunk(token)
                    if content:
                        modes_map["messages"].append(StepInfo(step="Token", message=content))
            elif stream_mode == "updates":
                # updates mode: yields dicts mapping node names to their state
                for node_name, data in chunk.items():
                    if "messages" in data and len(data["messages"]) > 0:
                        last_message = data["messages"][-1]
                        content = _render_completed_message(last_message)
                        modes_map["updates"].append(StepInfo(step=node_name, message=content))
                        # Track final response from AI messages
                        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                            final_response = content
            elif stream_mode == "custom":
                # custom mode: yields custom data from get_stream_writer()
                modes_map["custom"].append(StepInfo(step="", message=str(chunk)))

        # Convert map to list of StreamModeInfo
        stream_modes = [
            StreamModeInfo(stream_mode=mode, steps=steps)
            for mode, steps in modes_map.items()
            if steps
        ]

        return final_response, stream_modes  


class InvokeResponseHandler:
    """Handler for invoke (non-streaming) agent responses"""

    def handle_response(self, agent, messages: list) -> tuple[str, list[StreamModeInfo]]:
        result = agent.invoke({"messages": messages})

        # Handle different output structures
        if isinstance(result, dict):
            # Try common response keys
            response_text = (
                result.get("output") or result.get("messages", [])[-1].content
                if result.get("messages")
                else str(result)
            )
        else:
            response_text = str(result)

        return response_text, []  # No steps for invoke mode


class AgentQueryProcessor:
    """Processes agent queries with configurable response handler"""

    def __init__(self, response_handler: AgentResponseHandler):
        self.response_handler = response_handler

    async def process_query(self, query: str) -> QueryResponse:
        """Process a user query through the agent"""
        try:
            system_message = SystemMessage("You're a helpful assistant. Please answer questions by yourself or by the tool we provide to you.")
            human_message = HumanMessage(query)
            messages = [system_message, human_message]
            agent = get_agent()

            final_response, stream_modes = self.response_handler.handle_response(agent, messages)

            return QueryResponse(response=final_response, streamModes=stream_modes)
        except Exception as e:
            return QueryResponse(response="", error=str(e))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML page"""
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404, detail=f"index.html not found at {index_path}"
        )

    return FileResponse(index_path)


# Global processor instance - can be changed to switch response types
response_handler = StreamResponseHandler()
query_processor = AgentQueryProcessor(response_handler)


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Process user query through the agent"""
    return await query_processor.process_query(request.query)


@app.post("/api/query/invoke", response_model=QueryResponse)
async def query_agent_invoke(request: QueryRequest):
    """Process user query through the agent using invoke mode"""
    invoke_processor = AgentQueryProcessor(InvokeResponseHandler())
    return await invoke_processor.process_query(request.query)


@app.post("/api/query/stream", response_model=QueryResponse)
async def query_agent_stream(request: QueryRequest):
    """Process user query through the agent using stream mode"""
    stream_processor = AgentQueryProcessor(StreamResponseHandler())
    return await stream_processor.process_query(request.query)


def set_response_handler(handler: AgentResponseHandler):
    """Global function to change the response handler for the main /api/query endpoint"""
    global response_handler, query_processor
    response_handler = handler
    query_processor = AgentQueryProcessor(handler)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
