import express from "express";
import { ChatGroq } from "@langchain/groq";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import fs from "fs";
import { PDFParse } from "pdf-parse";
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

const upload = async () => {
  const pdfPath = "./knowledge.pdf";
  const buffer = await fs.readFileSync(pdfPath);
  const text = (await new PDFParse({ data: buffer }).getText()).text;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await splitter.createDocuments([text]);
  console.log(docs);
};

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
