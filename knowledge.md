# LangChain Agent Stream Method Response Structure

## `agent.stream()` Method Overview

The `agent.stream()` method returns a **generator** that yields chunks of data during the agent's execution. It allows real-time access to the agent's processing results.

### Basic Structure

```python
# Basic iteration pattern
for stream_mode, chunk in agent.stream({"messages": messages}, stream_mode=["messages"]):
    # Process the chunk
    pass
```

## Three Stream Modes

### 1. "messages" Mode 📝

**Purpose**: Streams individual message objects as they're generated
**Use Case**: Real-time display of the agent's response, token by token

**Structure**:
```python
for stream_mode, chunk in agent.stream({"messages": messages}, stream_mode=["messages"]):
    # chunk structure: (token, metadata)
    token, metadata = chunk

    # token is an AIMessageChunk object
    if isinstance(token, AIMessageChunk):
        # token has:
        # - text: str (partial response text)
        # - tool_call_chunks: List[ToolCallChunk] (partial tool calls)
        print(token.text)  # Display token-by-token
```

**Implementation**: Lines 136-147 in `app.py`

### 2. "custom" Mode 🎯

**Purpose**: Streams custom events and data defined by the developer
**Use Case**: Custom logging, debugging, or event tracking

**Structure**:
```python
for chunk in agent.stream({"messages": messages}, stream_mode=["custom"]):
    # chunk structure: Raw string data from get_stream_writer()
    # Example: "Looking up data for city: Tokyo"
    print(chunk)  # Custom progress events
```

**Implementation**: Lines 157-162 in `app.py`

### 3. "updates" Mode 🔍

**Purpose**: Streams detailed state updates from each node in the agent's execution graph
**Use Case**: Debugging and understanding the agent's internal execution flow

**Structure**:
```python
for chunk in agent.stream({"messages": messages}, stream_mode=["updates"]):
    # chunk structure: Dict mapping node names to their state
    # Example: {
    #     "agent": {"messages": [AIMessage, ...]},
    #     "tools": {"messages": [ToolMessage, ...]}
    # }

    for node_name, data in chunk.items():
        if "messages" in data:
            last_message = data["messages"][-1]
            print(f"{node_name}: {last_message.content}")
```

**Implementation**: Lines 172-183 in `app.py`

## Key Differences

| Mode | Structure | Best For | Performance |
|------|-----------|----------|-------------|
| **messages** | `(token, metadata)` tuples | Real-time UI | Lightweight |
| **custom** | Raw strings | Custom logging | Variable |
| **updates** | `{node: state}` dicts | Debugging | Heavyweight |

## Message Types Explained

### AIMessageChunk (from messages mode)
```python
AIMessageChunk(
    text="Hello",  # Partial response text
    tool_call_chunks=[...]  # Partial tool calls
)
```

### ToolMessage (from updates mode)
```python
ToolMessage(
    content="2026-02-10 15:55:58",  # Tool result
    tool_call_id="call_123"  # Which tool call
)
```

### AIMessage (final response from updates mode)
```python
AIMessage(
    content="The current date is February 10, 2026",  # Complete response
    tool_calls=[...]  # Called tools
)
```

## Best Practices

1. **Choose the right mode** for your use case:
   - Use "messages" for real-time chat UI
   - Use "custom" for debugging and custom events
   - Use "updates" for detailed execution analysis

2. **Handle different message types** appropriately (AIMessage, ToolMessage, etc.)

3. **Accumulate tokens** in messages mode to build the complete response

4. **Use metadata** when available for additional context

## Helper Functions in Implementation

### `_render_message_chunk(token: AIMessageChunk) -> str`
- Extracts text content from message chunks
- Filters out tool calls for cleaner output

### `_render_completed_message(message: AnyMessage) -> str`
- Renders different message types (AI, Tool, etc.)
- Provides detailed information about tool calls and results

## Stream Mode Handlers

Each stream mode has its own handler class:
- `MessagesModeHandler`: Processes token-by-token streaming
- `CustomModeHandler`: Handles custom events
- `UpdatesModeHandler`: Shows detailed debug information
- `StreamResponseHandler`: Aggregates multiple stream modes
- `InvokeResponseHandler`: Handles non-streaming responses

This structure allows for flexible and maintainable streaming implementation in LangChain agents.