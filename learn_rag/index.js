import express from "express";
import { ChatGroq } from "@langchain/groq";
import dotenv from "dotenv";
dotenv.config();
const app = express();
const port = 5000;

app.use(express.json());

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.7,
  maxTokens: undefined,
  maxRetries: 2,
  apiKey: process.env.GROQ_API_KEY,
});

app.post("/ai", async (req, res) => {
  const { input } = req.body;

  const response = await graph.invoke({
    messages: [
      {
        role: "user",
        content: input,
      },
    ],
  });
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
