import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

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


# Initialize Zhipu AI model


def get_agent():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("ZHIPUAI_API_KEY environment variable not set")
    llm = ChatZhipuAI(model="glm-4", api_key=api_key, temperature=0.7)

    agent = create_agent(llm, tools=[get_current_datetime])
    return agent


# define API, Request and Response
# TODO: pydantic?
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML page"""
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404, detail=f"index.html not found at {index_path}"
        )

    return FileResponse(index_path)


@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Process user query through the agent"""
    try:
        agent = get_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]}
        )
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

        return QueryResponse(response=response_text)
    except Exception as e:
        return QueryResponse(response="", error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
