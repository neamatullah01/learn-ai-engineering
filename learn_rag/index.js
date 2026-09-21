import express from "express";
import { ChatGroq } from "@langchain/groq";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { TaskType } from "@google/generative-ai";
import dotenv from "dotenv";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
dotenv.config();
const app = express();
const port = 5000;

app.use(express.json());

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0.7,
  maxTokens: undefined,
  maxRetries: 2,
});

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  collectionName: "grocery-rag-testing",
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
  const res = await vectorStore.addDocuments(docs);
};

app.post("/ai", async (req, res) => {
  const { input } = req.body;

  const docs = await vectorStore.similaritySearch(input, 5);
  const context = docs.map((d) => d.pageContent).join("/n");

  const response = await llm.invoke([
    new SystemMessage(`You are a RAG AI assistant.

STRICT RULES:
- Answer ONLY from context
- Do not use outside knowledge
- If answer not found say:
  "I don't know from uploaded PDF."

Context:
${context}`),
    new HumanMessage(input),
  ]);

  console.log(response);

  return res.status(200).json({ ai: response.content });
});

app.get("/", (req, res) => {
  return res.json({ message: "Hello from ai server" });
});

app.listen(port, () => {
  console.log("AI is Running");
});
