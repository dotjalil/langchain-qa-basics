import * as dotenv from "dotenv";
dotenv.config();

// import { OpenAI } from "langchain/llms/openai";
// import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

// const model = new OpenAI({
//   temperature: 0.5,
// });

const loader = new PuppeteerWebBaseLoader("https://iti.gov.eg/iti/home", {
  launchOptions: {
    headless: true,
  },
  gotoOptions: {
    waitUntil: "domcontentloaded",
    waitForSelector: ".sliderData",
  },
  /** Pass custom evaluate, in this case you get page and browser instances */
  async evaluate(page, browser) {
    console.log("Inside evaluate fn");
    console.log("page: ", page);
    await page.waitForSelector(".sliderData", { timeout: 20000 });

    const result = await page.evaluate(() => document.body.innerHTML);
    // console.log("result: ", result);
    return result;
  },
});

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
//   "What programs does ITI offer?"
// );

// console.log("vectorStore: ", vectorStore);
// console.log("relevantDocs: ", relevantDocs.length);

// 4

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0 });
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

const response = await chain.call({
  query:
    "ًWhat are the programs offered by ITI and give me a quick description about each program?",
});
console.log(response);
