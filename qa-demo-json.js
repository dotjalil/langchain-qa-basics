import * as dotenv from "dotenv";
dotenv.config();

import { OpenAI } from "langchain/llms/openai";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { JSONLoader } from "langchain/document_loaders/fs/json";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";

// const model = new OpenAI({
//   temperature: 0.5,
// });

const loader = new JSONLoader("data/facebook-posts.json");

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
  query: "ممكن تقولي معلومات اكتر عن مسار Furniture Design & Visualization",
});
console.log(response);

// { text: 'لا أعلم متى ستبدأ منحة التدريب المكثف.' }

/**
  query: "ممكن تقولي معلومات اكتر عن مسار Furniture Design & Visualization",
 * 
 */
// {
//   text: 'مسار Furniture Design & Visualization هو أحد المسارات المتاحة في معهد تكنولوجيا المعلومات، ويهدف إلى تدريب الطلاب على تصميم الأثاث واستخدام التقنيات المتقدمة في العرض البصري والتصور الثلاثي الأبعاد.\n' +
//     '\n' +
//     'هناك مجموعة من الخبراء في صناعة تصميم الأثاث والتصور الثلاثي الأبعاد يشاركون في تقييم أعمال الطلاب وتوجيههم، ومن بينهم م/عمرو عرنسة وم/هشام العيسوي ود/جيهان الدجوي.\n' +
//     '\n' +
//     'علاوة على ذلك، يوجد أيضًا فريق تدريس يضم م/هاجر حمدي وم/رنا ضياء، حيث يتولون مسؤولية تدريس وإشراف الطلاب على مجالات مختلفة مثل تصميم الألعاب وتطويرها وبرمجتها وبناء محاكاة الواقع الافتراضي والمعزز.\n' +
//     '\n' +
//     'لمزيد من التفاصيل حول المسار والتدريب المقدم، يمكنك مشاهدة الفيديو المرفق بالمنشور الترويجي لحفل تخريج الدفعة الجديدة.\n' +
//     '\n' +
//     'يرجى ملاحظة أن هذه المعلومات مقدمة بناءً على النص المقدم وقد تكون هناك معلومات إضافية تفصيلية غير مذكورة.'
// }
