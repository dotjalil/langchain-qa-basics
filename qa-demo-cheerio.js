import * as dotenv from "dotenv";
dotenv.config();

import { OpenAI } from "langchain/llms/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

// const model = new OpenAI({
//   temperature: 0.5,
// });

const loader = new CheerioWebBaseLoader(
  "https://ar.wikipedia.org/wiki/%D9%85%D8%B9%D9%87%D8%AF_%D8%AA%D9%83%D9%86%D9%88%D9%84%D9%88%D8%AC%D9%8A%D8%A7_%D8%A7%D9%84%D9%85%D8%B9%D9%84%D9%88%D9%85%D8%A7%D8%AA_(%D9%85%D8%B5%D8%B1)"
);

const data = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});

const splitDocs = await textSplitter.splitDocuments(data);

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// const relevantDocs = await vectorStore.similaritySearch(
//   "What is task decomposition?"
// );

// console.log(relevantDocs);

// 4

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo" });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.call({
  query: "What is ITI",
});
console.log(response);
