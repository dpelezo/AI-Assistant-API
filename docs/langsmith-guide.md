# LangSmith tracing

Tracing is optional. When it is on, each chat or research request sends a trace to [LangSmith](https://smith.langchain.com), so you can see every model call, tool call and its inputs and outputs.

## Turn it on

1. Create an API key in LangSmith under **Settings → API Keys**.
2. Set these environment variables:

   ```bash
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=ai-assistant-api
   ```

   Locally, put them in `.env`. On Vercel, add them under **Project → Settings → Environment Variables** and redeploy.
3. Start the API. The startup log should include `LangSmith tracing enabled for project: ai-assistant-api`.
4. Send a message, then open the project in LangSmith.

Prefer real environment variables over `.env` on any host. LangChain reads tracing settings from the process environment. The app copies the key, project and tracing flag from `.env` into the environment at startup, but not every hosting platform runs that startup step.

## What you will see

- **Agent runs.** LangChain and LangGraph trace each agent run automatically: the Claude calls (with the system prompt and messages) and each `retrieve_web_content` call to Exa.
- **Request runs.** The API also records its own runs, such as `Chat Query Processing` and `Chat Workflow Execution`, with the thread ID, the number of tool calls and any error. These appear as separate runs in the project, not nested under the agent run.

## Turn it off

Set `LANGSMITH_TRACING=false`, or remove `LANGSMITH_API_KEY`. The API then creates no runs and sends nothing to LangSmith.

## Privacy

Traces contain users' messages, search results and model answers. Do not turn tracing on for a public demo unless the demo page tells visitors that their messages are stored. Check your LangSmith plan's data retention period.

## Troubleshooting

- **No traces appear.** Check that `LANGSMITH_TRACING` is `true` and the key is set in the process environment, not only in `.env`.
- **An error log says `Error creating LangSmith run`.** The log includes the traceback. A wrong key or project name is the usual cause.
