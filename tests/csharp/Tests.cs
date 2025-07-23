using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading;
using UndreamAI.LlamaLib;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UndreamAI.LlamaLib.Tests
{
    [TestClass]
    public class LlamaLibTests
    {
        private const string PROMPT = "you are an artificial intelligence assistant\n\n### user: Hello, how are you?\n### assistant";
        private const int ID_SLOT = 0;
        private static readonly object completionLock = new object();
        private static int counter = 0;
        private static string concatData = "";
        // place model.gguf inside the tests folder
        private static string testModelPath => FindModel();

        public static string FindModel()
        {
            DirectoryInfo dir = new DirectoryInfo(Directory.GetCurrentDirectory());
            while (dir != null)
            {
                var candidate = Path.Combine(dir.FullName, "model.gguf");
                if (File.Exists(candidate))
                {
                    Console.WriteLine($"Found model: {candidate}");
                    return candidate;
                }

                dir = dir.Parent;
            }

            return null;
        }

        private static void CountCalls(string input)
        {
            concatData += input;
            counter++;
        }

        public void TestStart(LLMService llm)
        {
            Console.WriteLine("LLM_Start");
            Assert.IsTrue(llm.Start());
            Console.WriteLine("LLM_Started");
            Assert.IsTrue(llm.Started());
        }

        public void TestTemplate(LLM llm)
        {
            Console.WriteLine("LLM_Get_Template");
            Assert.AreEqual(llm.GetTemplate(), "chatml");

            Console.WriteLine("LLM_Apply_Template");
            var messages = new JArray();
            messages.Add(new JObject{["role"] = "user", ["content"] = "how are you?"});
            messages.Add(new JObject{["role"] = "assistant", ["content"] = "fine, thanks, and you?"});

            string messagesFormatted = llm.ApplyTemplate(messages);
            string messagesFormattedGT = "<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\nfine, thanks, and you?";
            Assert.AreEqual(messagesFormatted, messagesFormattedGT);
        }

        public void TestTokenization(LLM llm)
        {
            Console.WriteLine("LLM_Tokenize");
            List<int> tokens = llm.Tokenize(PROMPT);
            Assert.IsTrue(tokens.Count > 0);

            Console.WriteLine("LLM_Detokenize");
            string detokenizedContent = llm.Detokenize(tokens);
            Assert.AreEqual(PROMPT.Trim(), detokenizedContent.Trim());
        }
        
        private void TestCompletionWithStreaming(LLM llm, bool stream)
        {
            Console.WriteLine($"LLM_Completion ({(stream ? "" : "no ")}streaming)");

            lock (completionLock)
            {
                counter = 0;
                concatData = "";
                string content = llm.Completion(PROMPT, stream? CountCalls: null, ID_SLOT, null);
                Assert.IsFalse(string.IsNullOrEmpty(content));
                if (stream)
                {
                    Assert.IsTrue(counter > 0);
                    Assert.IsTrue(content == concatData);
                }
            }
        }
        
        public void TestCompletion(LLM llm)
        {
            TestCompletionWithStreaming(llm, false);
            TestCompletionWithStreaming(llm, true);
        }
        
        public void TestEmbedding(LLM llm, int embeddingSize)
        {
            Console.WriteLine("LLM_Embeddings");

            List<float> embedding = llm.Embeddings(PROMPT);
            Assert.AreEqual(embeddingSize, embedding.Count);
        }

        public void TestSetTemplate(LLMProvider llm)
        {
            Console.WriteLine("LLM_Set_Template");
            llm.SetTemplate("phi3");
            Console.WriteLine("LLM_Get_Template");
            Assert.AreEqual("phi3", llm.GetTemplate());
            llm.SetTemplate("chatml");
        }

        public void TestLoraList(LLMProvider llm)
        {
            Console.WriteLine("LLM_Lora_List");
            List<LoraIdScalePath> loras = llm.LoraList();
            Assert.AreEqual(0, loras.Count);
        }

        public void TestSlotSaveRestore(LLMLocal llm)
        {
            Console.WriteLine("LLM_Slot Save");

            string filename = "test_undreamai.save";
            string filepath = Path.Combine(Directory.GetCurrentDirectory(), filename);

            string saveResult = llm.Slot(ID_SLOT, "save", filepath);

            Assert.AreEqual(filename, saveResult);
            Assert.IsTrue(File.Exists(filepath));

            Console.WriteLine("LLM_Slot Restore");

            string restoreResult = llm.Slot(ID_SLOT, "restore", filepath);

            Assert.AreEqual(filename, restoreResult);
            File.Delete(filepath);
        }

        public void TestCancel(LLMLocal llm)
        {
            Console.WriteLine("LLM_Cancel");
            llm.Cancel(ID_SLOT);
        }

        [TestMethod]
        public void Tests_LLMService()
        {
            LLMService llmService = new LLMService(testModelPath);
            llmService.numPredict = 10;

            TestStart(llmService);
            TestTemplate(llmService);
            TestSetTemplate(llmService);
            TestTokenization(llmService);
            TestCompletion(llmService);
            TestEmbedding(llmService, llmService.EmbeddingSize());
            TestSlotSaveRestore(llmService);
            TestLoraList(llmService);
            TestCancel(llmService);
            
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMClient()
        {
            LLMService llmService = LLMService.FromCommand(new String("-m " + testModelPath));
            TestStart(llmService);

            LLMClient llmClient = new LLMClient(llmService);
            llmClient.numPredict = 10;

            TestTemplate(llmClient);
            TestTokenization(llmClient);
            TestCompletion(llmClient);
            TestEmbedding(llmClient, llmService.EmbeddingSize());
            TestSlotSaveRestore(llmClient);
            TestCancel(llmClient);

            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMRemoteClient()
        {
            LLMService llmService = new LLMService(testModelPath);
            TestStart(llmService);
            llmService.StartServer("", 13333);

            LLMClient llmClient = new LLMClient("http://localhost", 13333);
            llmClient.numPredict = 10;

            TestTemplate(llmService);
            TestTokenization(llmClient);
            TestCompletion(llmClient);
            TestEmbedding(llmClient, llmService.EmbeddingSize());

            llmService?.Dispose();
        }
    }
}