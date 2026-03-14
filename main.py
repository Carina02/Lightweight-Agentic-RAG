import os
import operator
import json
from typing import TypedDict, Annotated, List, Literal
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ====== 0. Environment & Slack Setup ======
# Initialize Slack WebClient using the Bot Token from environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "your-xoxb-token-here")
slack_client = WebClient(token=SLACK_BOT_TOKEN)
TARGET_SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL_ID", "C0123456789")  # Replace with actual channel ID


def send_slack_approval_request(thread_id: str, question: str, generation: str, channel_id: str):
    """Construct and send a Slack Block Kit message for human-in-the-loop approval."""
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "⚠️ Human Intervention Required: Low Confidence Answer",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original Question:*\n{question}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Generated Answer (Needs Review):*\n```\n{generation}\n```"}
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve", "emoji": True},
                    "style": "primary",
                    "value": f"Approve|{thread_id}",
                    "action_id": "approve_action"
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject", "emoji": True},
                    "style": "danger",
                    "value": f"Reject|{thread_id}",
                    "action_id": "reject_action"
                }
            ]
        }
    ]
    try:
        response = slack_client.chat_postMessage(
            channel=channel_id,
            blocks=blocks,
            text="Human intervention required for RAG workflow."
        )
        print(f"--> [Slack API] Message successfully sent to {channel_id}. Timestamp: {response['ts']}")
    except SlackApiError as e:
        print(f"--> [Slack API Error]: {e.response['error']}")


# ====== 1. State Definition ======
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: Annotated[int, operator.add]
    status: Literal["PASS", "FAIL", "PENDING"]
    route: str


# ====== 2. Node Logic ======
def retrieve_node(state: GraphState):
    # Retrieve documents via ChromaDB or SQLite
    return {"documents": ["retrieved_doc_1", "retrieved_doc_2"]}


def generate_node(state: GraphState):
    # Generate answer using GPT-4o-mini
    return {"generation": "Generated answer from 4o-mini based on retrieved context."}


def critic_node(state: GraphState):
    # Evaluate hallucination and relevance using GPT-4o
    # Simulating a failure scenario here to trigger the loop/human intervention
    is_valid = False

    if is_valid:
        return {"status": "PASS"}
    else:
        return {"status": "FAIL", "retries": 1}


def rewrite_node(state: GraphState):
    # Rewrite the query for better retrieval
    return {"question": state["question"] + " (optimized)"}


def human_node(state: GraphState, config: dict):
    thread_id = config["configurable"]["thread_id"]

    # Send Block Kit message to Slack
    send_slack_approval_request(
        thread_id=thread_id,
        question=state["question"],
        generation=state["generation"],
        channel_id=TARGET_SLACK_CHANNEL
    )

    print(f"--> [Workflow] Thread {thread_id} suspended. Waiting for Slack Interaction...")

    # Suspend execution graph
    human_decision = interrupt("Waiting for Slack Approval")

    if human_decision == "Approve":
        return {"generation": state["generation"] + "\n\n[Status: Approved by Human]"}
    else:
        return {"generation": "[Status: Rejected by Human. Workflow Terminated.]"}


# ====== 3. Routing Edge Logic ======
def route_after_critic(state: GraphState) -> str:
    if state["status"] == "PASS":
        return END
    elif state["retries"] >= 3:
        return "human_node"  # Trigger human intervention after 3 failed retries
    else:
        return "rewrite_node"


# ====== 4. Graph Compilation ======
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("critic", critic_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("human_node", human_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critic")
workflow.add_conditional_edges("critic", route_after_critic)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("human_node", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# ====== 5. FastAPI Gateway ======
app = FastAPI()


class QueryRequest(BaseModel):
    thread_id: str
    question: str


@app.post("/api/ask")
async def start_workflow(req: QueryRequest, background_tasks: BackgroundTasks):
    """Trigger the LangGraph workflow."""
    config = {"configurable": {"thread_id": req.thread_id}}

    def run_graph():
        for event in app_graph.stream({"question": req.question, "retries": 0, "status": "PENDING"}, config):
            pass

    background_tasks.add_task(run_graph)
    return {"status": "Workflow triggered", "thread_id": req.thread_id}


@app.post("/slack/interactions")
async def slack_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive Slack Interactive Button callbacks.
    Must return HTTP 200 within 3 seconds to avoid Slack timeout errors.
    """
    form_data = await request.form()
    payload = json.loads(form_data.get("payload", "{}"))

    action_value = payload.get("actions", [{}])[0].get("value", "")

    if "|" not in action_value:
        return {"error": "Invalid action value format"}

    decision, thread_id = action_value.split("|")
    config = {"configurable": {"thread_id": thread_id}}

    # Isolate graph resumption into a background task to immediately free the HTTP response
    def resume_graph():
        print(f"--> [Webhook] Resuming thread {thread_id} with decision: {decision}")
        app_graph.invoke(Command(resume=decision), config=config)

    background_tasks.add_task(resume_graph)

    return {"text": f"Received human intervention command: {decision}"}


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)