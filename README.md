# Lightweight Agentic RAG with Human-in-the-Loop (HITL)

A pure Python, minimal-dependency Agentic RAG workflow designed for complex unstructured document processing. 

This system replaces heavy automation platforms (like n8n) with a code-first approach using **LangGraph** for state management and **FastAPI** for API routing. It implements a multi-agent validation loop with integrated **Slack Block Kit** for asynchronous Human-in-the-Loop (HITL) approval when the LLM confidence threshold is not met.

## Architecture & Logic



1. **Trigger & Route:** FastAPI receives the initial query. The router agent determines the data source (ChromaDB for unstructured text, SQLite for structured data).
2. **Retrieve & Generate:** Fetches context and generates a preliminary answer using a cost-efficient model (e.g., GPT-4o-mini).
3. **Answer Critic (Dual Validation):** A reasoning model (e.g., GPT-4o) evaluates the answer based on two strict criteria:
   - **Hallucination Check:** Is the answer strictly grounded in the retrieved documents?
   - **Relevance Check:** Does the answer directly address the user's original query?
4. **Autonomous Rewrite & Loop:** If the critic fails, the system rewrites the query and re-retrieves. Maximum loop limit is strictly set to 3.
5. **Human-in-the-Loop (Slack Integration):** If the loop limit is reached and quality remains below the threshold, the LangGraph execution is suspended via `interrupt()`. A Block Kit message is pushed to Slack. The graph resumes computation upon receiving the `Approve` or `Reject` webhook from the Slack Interactive button.

## 🛠 Tech Stack

- **Orchestration:** LangGraph (StateGraph, MemorySaver)
- **API Gateway:** FastAPI, Uvicorn
- **LLM Integration:** LangChain, OpenAI API
- **Vector / Relational DB:** ChromaDB (Local), SQLite (Native)
- **Notifications:** Slack SDK (`slack_sdk`)

## ⚙️ Prerequisites & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/agentic-rag-hitl.git](https://github.com/yourusername/agentic-rag-hitl.git)
cd agentic-rag-hitl
```
**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Environment Variables**
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=sk-your-openai-api-key
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
```
**4. Deployment & Execution**
Run the FastAPI server locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Note: For local testing with Slack webhooks, use `ngrok` to expose your local port:
```bash
ngrok http 8000
```
## 🔗 API Endpoints
**1. Start Workflow
POST `/api/ask`
Triggers the Agentic RAG background task.

Request Body:
```bash
{
  "thread_id": "req-99812",
  "question": "Analyze the revenue recognition clauses in the Q3 financial report."
}
```
Response:
```bash
{
  "status": "Workflow triggered",
  "thread_id": "req-99812"
}
```



**2. Slack Interaction Webhook
POST `/slack/interactions`
Endpoint configured in the Slack App dashboard to receive Interactive Button callbacks. It extracts the `thread_id` and the human decision `(Approve/Reject)` to resume the suspended LangGraph state.

**3. Slack App Configuration Guide
To enable the Human-in-the-Loop functionality, your Slack App must be configured correctly:

1. Go to Slack API Dashboard -> Your App.

2. Under OAuth & Permissions, add the `chat:write` scope. Install the app to your workspace and copy the Bot User OAuth Token `(xoxb-...)`.

3. Under Interactivity & Shortcuts, toggle Interactivity to On.

4. Set the Request URL to your server's public endpoint: `https://<your-domain-or-ngrok>/slack/interactions`.

5. Invite the bot to your target channel: `/invite @YourBotName`.

## 📄 License
MIT License
