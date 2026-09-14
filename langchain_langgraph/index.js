import express from "express";
import { ChatGroq } from "@langchain/groq";
import {
  MemorySaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import dotenv from "dotenv";
dotenv.config();
const app = express();
const port = 5000;

app.use(express.json());

const webSearchTool = new TavilySearch({
  maxResults: 5,
  topic: "general",
});

const checkPointer = new MemorySaver();

const tools = [webSearchTool];
const toolNode = new ToolNode(tools);

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.7,
  maxTokens: undefined,
  maxRetries: 2,
  apiKey: process.env.GROQ_API_KEY,
}).bindTools(tools);

const callLLM = async (state) => {
  console.log("state:", state);
  const response = await llm.invoke([
    {
      role: "system",
      content: `You are Jarvis AI assistant

Use conversation memory first.

Only use tools when the answer requires
external real-time information like:
weather, news, web search, stock prices etc.

Do NOT call tools for simple conversation,
memory-based questions, greetings,
or personal context`,
    },
    ...state.messages,
  ]);
  return { messages: [response] };
};

const shouldContinue = async (state) => {
  const lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage.tool_calls.length > 0) {
    return "tools";
  } else {
    return "__end__";
  }
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callLLM)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .compile({ checkpointer: checkPointer });

app.post("/ai", async (req, res) => {
  const { input } = req.body;

  const response = await graph.invoke(
    {
      messages: [
        {
          role: "user",
          content: input,
        },
      ],
    },
    { configurable: { thread_id: "user123" } },
  );
  return res
    .status(200)
    .json({ "ai:": response.messages[response.messages.length - 1].content });
});

app.get("/", (req, res) => {
  return res.json({ message: "Hello from ai server" });
});

app.listen(port, () => {
  console.log("AI is Running");
});
