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
        private const string SYSTEM_PROMPT = "you are an artificial intelligence assistant";
        private const string PROMPT = $"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nhow are you?<|im_end|>\n<|im_start|>assistant\n";
        private const int ID_SLOT = 0;
        private static readonly object completionLock = new object();
        private static int counter = 0;
        private static string concatData = "";
        // place model.gguf inside the tests folder
        private static string testModelPath => FindModel("model.gguf");
        private static string testModelEmbeddingPath => FindModel("model_embedding.gguf");
        private int EMBEDDING_SIZE = 0;

        public static string FindModel(string modelName)
        {
            DirectoryInfo dir = new DirectoryInfo(Directory.GetCurrentDirectory());
            while (dir != null)
            {
                var candidate = Path.Combine(dir.FullName, modelName);
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
            concatData = input;
            counter++;
        }

        public void TestStart(LLMService llm)
        {
            Console.WriteLine("LLM_Start");
            Assert.IsTrue(llm.Start());
            Console.WriteLine("LLM_Started");
            Assert.IsTrue(llm.Started());
            EMBEDDING_SIZE = llm.EmbeddingSize();
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
            llm.SetCompletionParameters(new JObject { ["n_predict"] = 30 });
            Console.WriteLine($"LLM_Completion ({(stream ? "" : "no ")}streaming)");

            lock (completionLock)
            {
                counter = 0;
                concatData = "";
                string content = llm.Completion(PROMPT, stream ? CountCalls : null, ID_SLOT);
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

        public void TestEmbedding(LLM llm)
        {
            Console.WriteLine("LLM_Embeddings");

            List<float> embedding = llm.Embeddings(PROMPT);
            Assert.AreEqual(EMBEDDING_SIZE, embedding.Count);
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

            string saveResult = llm.SaveSlot(ID_SLOT, filepath);

            Assert.AreEqual(filename, saveResult);
            Assert.IsTrue(File.Exists(filepath));

            Console.WriteLine("LLM_Slot Restore");

            string restoreResult = llm.LoadSlot(ID_SLOT, filepath);

            Assert.AreEqual(filename, restoreResult);
            File.Delete(filepath);
        }

        public void TestCancel(LLMLocal llm)
        {
            Console.WriteLine("LLM_Cancel");
            llm.Cancel(ID_SLOT);
        }

        public void TestLLMAgentProperties(LLMAgent agent)
        {
            agent.SystemPrompt = "New system prompt";
            Assert.AreEqual("New system prompt", agent.SystemPrompt);

            // Test SlotId property
            int originalSlot = agent.SlotId;
            Assert.IsTrue(originalSlot >= 0);

            agent.SlotId = originalSlot + 1;
            Assert.AreEqual(originalSlot + 1, agent.SlotId);

            agent.SystemPrompt = SYSTEM_PROMPT;
            agent.SlotId = originalSlot;
        }

        public void TestHistoryManagement(LLMAgent agent)
        {
            Console.WriteLine("History Management Tests");

            Assert.AreEqual(0, agent.GetHistorySize());

            // Test adding messages
            agent.AddUserMessage("Test user message");
            agent.AddAssistantMessage("Test assistant response");

            Assert.AreEqual(2, agent.GetHistorySize());
            var history = agent.GetHistory();
            Assert.AreEqual(2, history.Count);
            Assert.AreEqual("user", history[0].role);
            Assert.AreEqual("Test user message", history[0].content);
            Assert.AreEqual("assistant", history[1].role);
            Assert.AreEqual("Test assistant response", history[1].content);

            // Test AddUserMessage and AddAssistantMessage
            agent.AddUserMessage("User message via shortcut");
            agent.AddAssistantMessage("Assistant message via shortcut");

            Assert.AreEqual(4, agent.GetHistorySize());
            history = agent.GetHistory();
            Assert.AreEqual("User message via shortcut", history[2].content);
            Assert.AreEqual("Assistant message via shortcut", history[3].content);

            // Test removing last message
            agent.RemoveLastMessage();
            Assert.AreEqual(3, agent.GetHistorySize());

            // Test clearing history
            agent.ClearHistory();
            Assert.AreEqual(0, agent.GetHistorySize()); // Should keep system prompt

            // Create test history
            var testMessages = new List<ChatMessage>
            {
                new ChatMessage("system", "Test system"),
                new ChatMessage("user", "Hello"),
                new ChatMessage("assistant", "Hi there!"),
                new ChatMessage("user", "How are you?"),
                new ChatMessage("assistant", "I'm doing well, thanks!")
            };

            // Test SetHistory
            agent.SetHistory(testMessages);
            Assert.AreEqual(testMessages.Count, agent.GetHistorySize());

            var retrievedHistory = agent.GetHistory();
            Assert.AreEqual(testMessages.Count, retrievedHistory.Count);
            for (int i = 0; i < testMessages.Count; i++)
            {
                Assert.AreEqual(testMessages[i], retrievedHistory[i]);
            }

            // Test History property getter/setter
            var historyJson = agent.History;
            Assert.IsTrue(historyJson.Count == testMessages.Count);

            var newHistoryJson = new JArray();
            newHistoryJson.Add(new JObject { ["role"] = "system", ["content"] = "New system" });
            newHistoryJson.Add(new JObject { ["role"] = "user", ["content"] = "New user message" });

            agent.History = newHistoryJson;
            Assert.AreEqual(2, agent.GetHistorySize());

            // Test null handling
            Assert.ThrowsException<ArgumentNullException>(() => agent.SetHistory(null));
        }

        public void TestHistoryFileOperations(LLMAgent agent)
        {
            Console.WriteLine("History File Operations Tests");
            string filename = "test_agent_history.json";
            string filepath = Path.Combine(Directory.GetCurrentDirectory(), filename);

            try
            {
                // Add some test messages
                agent.AddUserMessage("Test message 1");
                agent.AddAssistantMessage("Test response 1");
                agent.AddUserMessage("Test message 2");
                agent.AddAssistantMessage("Test response 2");
                int originalSize = agent.GetHistorySize();

                // Test saving to file
                agent.SaveHistory(filepath);
                Assert.IsTrue(File.Exists(filepath));

                // Clear history and verify
                agent.ClearHistory();
                Assert.AreEqual(0, agent.GetHistorySize());

                // Test loading from file
                agent.LoadHistory(filepath);
                Assert.AreEqual(originalSize, agent.GetHistorySize());

                // Test argument validation
                Assert.ThrowsException<ArgumentNullException>(() => agent.SaveHistory(null));
                Assert.ThrowsException<ArgumentNullException>(() => agent.SaveHistory(""));
                Assert.ThrowsException<ArgumentNullException>(() => agent.LoadHistory(null));
                Assert.ThrowsException<ArgumentNullException>(() => agent.LoadHistory(""));
            }
            finally
            {
                if (File.Exists(filepath))
                    File.Delete(filepath);
            }
        }

        public void TestChatFunctionality(LLMAgent agent)
        {
            Console.WriteLine("Chat Functionality Tests");

            agent.ClearHistory();

            string userPrompt = "Hello, how are you?";

            // Test chat with history addition
            string response1 = agent.Chat(userPrompt, addToHistory: true);
            Assert.IsFalse(string.IsNullOrEmpty(response1));
            Assert.AreEqual(2, agent.GetHistorySize());

            var history = agent.GetHistory();
            Assert.AreEqual("user", history[0].role);
            Assert.AreEqual(userPrompt, history[0].content);
            Assert.AreEqual("assistant", history[1].role);
            Assert.AreEqual(response1, history[1].content);

            // Test chat without history addition
            string response2 = agent.Chat("Another question", addToHistory: false);
            Assert.IsFalse(string.IsNullOrEmpty(response2));
            Assert.AreEqual(2, agent.GetHistorySize());
        }

        public void LLMTests(LLM llm)
        {
            TestTokenization(llm);
            TestCompletion(llm);
        }

        public void LLMLocalTests(LLMLocal llm)
        {
            LLMTests(llm);
            TestSlotSaveRestore(llm);
            TestCancel(llm);
        }

        public void LLMProviderTests(LLMProvider llm)
        {
            LLMLocalTests(llm);
            TestLoraList(llm);
        }

        public void LLMAgentTests(LLMAgent llm)
        {
            LLMLocalTests(llm);
            TestLLMAgentProperties(llm);
            TestHistoryManagement(llm);
            TestHistoryFileOperations(llm);
            TestChatFunctionality(llm);
        }


        [TestMethod]
        public void Tests_LLMService()
        {
            LLMService llmService = new LLMService(testModelPath);
            TestStart(llmService);
            LLMProviderTests(llmService);
            llmService?.Dispose();

            llmService = new LLMService(testModelEmbeddingPath, embeddingOnly: true);
            TestStart(llmService);
            TestEmbedding(llmService);
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMClient()
        {
            LLMService llmService = LLMService.FromCommand(new String("-m " + testModelPath));
            TestStart(llmService);

            LLMClient llmClient = new LLMClient(llmService);
            LLMLocalTests(llmClient);
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMRemoteClient()
        {
            LLMService llmService = new LLMService(testModelPath);
            TestStart(llmService);
            llmService.StartServer("", 13333);

            LLMClient llmClient = new LLMClient("http://localhost", 13333);
            LLMTests(llmClient);
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMAgent()
        {
            LLMService llmService = new LLMService(testModelPath);
            TestStart(llmService);

            LLMAgent llmAgent = new LLMAgent(llmService, SYSTEM_PROMPT);
            LLMAgentTests(llmAgent);
            llmService?.Dispose();
        }
    }
}