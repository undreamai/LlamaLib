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
        string testModelPath = "../../../../model.gguf";

        private static void CountCalls(string input)
        {
            concatData += input;
            counter++;
        }

        public void TestStart(Func<bool> startAction, Func<bool> startedAction)
        {
            Console.WriteLine("LLM_Start");
            Assert.IsTrue(startAction());
            Console.WriteLine("LLM_Started");
            Assert.IsTrue(startedAction());
        }

        public void TestTokenization(Func<string, List<int>> tokenizeFunc, Func<List<int>, string> detokenizeFunc)
        {
            Console.WriteLine("LLM_Tokenize");
            List<int> tokens = tokenizeFunc(PROMPT);
            Assert.IsTrue(tokens.Count > 0);

            Console.WriteLine("LLM_Detokenize");
            string detokenizedContent = detokenizeFunc(tokens);
            Assert.AreEqual(PROMPT.Trim(), detokenizedContent.Trim());
        }
        
        private void TestCompletionWithStreaming(Func<string, int, LlamaLib.CharArrayCallback, JObject, string> completionFunc, bool stream)
        {
            Console.WriteLine($"LLM_Completion ({(stream ? "" : "no ")}streaming)");

            lock (completionLock)
            {
                counter = 0;
                concatData = "";
                string content = completionFunc(PROMPT, ID_SLOT, stream? CountCalls: null, null);
                Assert.IsFalse(string.IsNullOrEmpty(content));
                if (stream)
                {
                    Assert.IsTrue(counter > 0);
                    Assert.IsTrue(content == concatData);
                }
            }
        }
        
        public void TestCompletion(Func<string, int, LlamaLib.CharArrayCallback, JObject, string> completionFunc)
        {
            TestCompletionWithStreaming(completionFunc, false);
            TestCompletionWithStreaming(completionFunc, true);
        }
        
        public void TestEmbedding(Func<string, List<float>> embeddingFunc, Func<int> embeddingSizeFunc)
        {
            Console.WriteLine("LLM_Embeddings");

            List<float> embedding = embeddingFunc(PROMPT);
            Assert.AreEqual(embeddingSizeFunc(), embedding.Count);
        }

        public void TestLoraList(Func<List<LoraIdScalePath>> loraListFunc)
        {
            Console.WriteLine("LLM_Lora_List");
            List<LoraIdScalePath> loras = loraListFunc();
            Assert.AreEqual(0, loras.Count);
        }

        public void TestSlotSaveRestore(Func<int, string, string, string> slotFunc)
        {
            Console.WriteLine("LLM_Slot Save");

            string filename = "test_undreamai.save";
            string filepath = Path.Combine(Directory.GetCurrentDirectory(), filename);

            string saveResult = slotFunc(ID_SLOT, "save", filepath);

            Assert.AreEqual(filename, saveResult);
            Assert.IsTrue(File.Exists(filepath));

            Console.WriteLine("LLM_Slot Restore");

            string restoreResult = slotFunc(ID_SLOT, "restore", filepath);

            Assert.AreEqual(filename, restoreResult);
            File.Delete(filepath);
        }

        public void TestCancel(Action<int> cancelAction)
        {
            Console.WriteLine("LLM_Cancel");
            cancelAction(ID_SLOT);
        }

        [TestMethod]
        public void Tests_LLMService()
        {
            LLMService llmService = new LLMService(testModelPath);

            TestStart(llmService.Start, llmService.Started);
            TestTokenization(llmService.Tokenize, llmService.Detokenize);
            TestCompletion(llmService.Completion);
            TestEmbedding(llmService.Embeddings, llmService.EmbeddingSize);
            TestSlotSaveRestore(llmService.Slot);
            TestLoraList(llmService.LoraList);
            TestCancel(llmService.Cancel);
            
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMClient()
        {
            LLMService llmService = LLMService.FromCommand(new String("-m " + testModelPath));
            TestStart(llmService.Start, llmService.Started);

            LLMClient llmClient = new LLMClient(llmService);
            llmClient.numPredict = 10;

            TestTokenization(llmClient.Tokenize, llmClient.Detokenize);
            TestCompletion(llmClient.Completion);
            TestEmbedding(llmClient.Embeddings, llmService.EmbeddingSize);
            TestSlotSaveRestore(llmClient.Slot);
            TestCancel(llmClient.Cancel);

            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMRemoteClient()
        {
            LLMService llmService = new LLMService(testModelPath);
            TestStart(llmService.Start, llmService.Started);
            llmService.StartServer("", 13333);

            LLMRemoteClient llmClient = new LLMRemoteClient("http://localhost", 13333);
            llmClient.numPredict = 10;

            TestTokenization(llmClient.Tokenize, llmClient.Detokenize);
            TestCompletion(llmClient.Completion);
            TestEmbedding(llmClient.Embeddings, llmService.EmbeddingSize);

            llmService?.Dispose();
        }
    }
}