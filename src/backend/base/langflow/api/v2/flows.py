from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["Flows"], prefix="/flows")


# Pydantic model for the request
class FlowRequest(BaseModel):
    query: str

# Component templates
def generate_node_template(
    node_type: str,
    display_name: str,
    inputs: list[str],
    outputs: list[str],
    config: dict | None = None
):
    """Generate a generic node template."""
    return {
        "data": {
            "description": f"{display_name} node.",
            "display_name": display_name,
            "type": node_type,
            "node": {
                "base_classes": [node_type],
                "inputs": [{"name": inp, "types": ["Message"]} for inp in inputs],
                "outputs": [{"name": outp, "types": ["Message"]} for outp in outputs],
                "template": config or {}
            }
        },
        "type": "genericNode",
        "id": f"{node_type}-{str(uuid.uuid4())[:8]}"  # Unique ID for the node
    }

def generate_edge(source_id: str, target_id: str, source_handle: str, target_handle: str):
    """Generate an edge between two nodes."""
    return {
        "source": source_id,
        "target": target_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle
    }

# Parse the query and map to components
def parse_query(query: str) -> list[dict]:
    """Parse the natural language query and map it to components."""
    # Example mapping (extend this based on your system's components)
    component_mapping = {
        "string": {"type": "TextInput", "display_name": "Text Input", "outputs": ["text_output"]},
        "embedding": {
            "type": "EmbeddingModel",
            "display_name": "Embedding Model",
            "inputs": ["text_input"],
            "outputs": ["embedding"]
        },
        "chat": {"type": "ChatInput", "display_name": "Chat Input", "outputs": ["message"]},
        "openai": {
            "type": "OpenAIModel",
            "display_name": "OpenAI Model",
            "inputs": ["message"],
            "outputs": ["text_output"]
        },
        "output": {"type": "ChatOutput", "display_name": "Chat Output", "inputs": ["message"]}
    }

    # Extract components from the query (this is a simple example; use NLP for better parsing)
    components = []
    if "string" in query:
        components.append(component_mapping["string"])
    if "embedding" in query:
        components.append(component_mapping["embedding"])
    if "chat" in query:
        components.append(component_mapping["chat"])
    if "openai" in query:
        components.append(component_mapping["openai"])
    if "output" in query:
        components.append(component_mapping["output"])

    return components

# Generate the flow data
def generate_flow_data(query: str) -> dict:
    """Generate the `data` object for the flow based on the query."""
    components = parse_query(query)

    if not components:
        msg = "No valid components found in the query."
        raise ValueError(msg)

    # Generate nodes
    nodes = []
    for component in components:
        node = generate_node_template(
            node_type=component["type"],
            display_name=component["display_name"],
            inputs=component.get("inputs", []),
            outputs=component.get("outputs", [])
        )
        nodes.append(node)

    # Generate edges
    edges = []
    for i in range(len(nodes) - 1):
        source_node = nodes[i]
        target_node = nodes[i + 1]
        source_handle = source_node["data"]["node"]["outputs"][0]["name"]
        target_handle = target_node["data"]["node"]["inputs"][0]["name"]
        edge = generate_edge(
            source_id=source_node["id"],
            target_id=target_node["id"],
            source_handle=source_handle,
            target_handle=target_handle
        )
        edges.append(edge)

    return {
        "nodes": nodes,
        "edges": edges
    }

# FastAPI endpoint
@router.post("/", response_model=dict)
async def create_flow(request: FlowRequest):
    try:
        return generate_flow_data(request.query)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
