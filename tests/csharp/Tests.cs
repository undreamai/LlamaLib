using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
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
        private static int counter = 0;
        string testModelPath = "/home/benuix/.config/LLMUnity/models/smol_llama-220m-openhermes.q4_k_m.gguf";

        private static string ConcatenateStreamingResult(string input)
        {
            var lines = input.Split('\n');
            var output = "";

            foreach (var line in lines)
            {
                if (line.StartsWith("data: "))
                {
                    var jsonStr = line.Substring(6);
                    try
                    {
                        var parsed = JsonSerializer.Deserialize<Dictionary<string, object>>(jsonStr);
                        if (parsed.ContainsKey("content"))
                        {
                            output += parsed["content"].ToString();
                        }
                    }
                    catch (JsonException ex)
                    {
                        Console.WriteLine($"JSON parse error: {ex.Message}");
                    }
                }
            }
            return output;
        }

        private static void CountCalls(IntPtr charArray)
        {
            counter++;
        }

        public void TestStart(Action startAction, Func<bool> startedAction)
        {
            Console.WriteLine("LLM_Start");
            startAction();
            Console.WriteLine("LLM_Started");
            Assert.IsTrue(startedAction());
        }

        public void TestTokenization(Func<string, string> tokenizeFunc, Func<string, string> detokenizeFunc)
        {
            Console.WriteLine("LLM_Tokenize");

            var data = new Dictionary<string, object> { ["content"] = PROMPT };
            var json = JsonSerializer.Serialize(data);

            var reply = tokenizeFunc(json);
            var replyData = JsonSerializer.Deserialize<Dictionary<string, object>>(reply);
            Assert.IsTrue(replyData.ContainsKey("tokens"));

            var tokens = (JsonElement)replyData["tokens"];
            Assert.IsTrue(tokens.GetArrayLength() > 0);

            Console.WriteLine("LLM_Detokenize");

            var detokenized = detokenizeFunc(reply);
            var detokenizedData = JsonSerializer.Deserialize<Dictionary<string, object>>(detokenized);
            Assert.AreEqual(PROMPT.Trim(), detokenizedData["content"].ToString().Trim());
        }

        private void TestCompletionWithStreaming(Func<string, LlamaLib.CharArrayCallback, string> completionFunc, bool stream)
        {
            Console.WriteLine($"LLM_Completion ({(stream ? "" : "no ")}streaming)");

            var data = new Dictionary<string, object>
            {
                ["id_slot"] = ID_SLOT,
                ["prompt"] = PROMPT,
                ["cache_prompt"] = true,
                ["n_predict"] = 10,
                ["n_keep"] = 30,
                ["stream"] = stream
            };

            counter = 0;
            var json = JsonSerializer.Serialize(data);
            var reply = completionFunc(json, CountCalls);

            string content = stream ? ConcatenateStreamingResult(reply) : JsonSerializer.Deserialize<Dictionary<string, object>>(reply)["content"].ToString();
            Assert.IsFalse(string.IsNullOrEmpty(content));
        }

        public void TestCompletion(Func<string, LlamaLib.CharArrayCallback, string> completionFunc)
        {
            TestCompletionWithStreaming(completionFunc, false);
            TestCompletionWithStreaming(completionFunc, true);
        }

        public void TestEmbedding(Func<string, string> embeddingFunc, Func<int> embeddingSizeFunc)
        {
            Console.WriteLine("LLM_Embeddings");

            var data = new Dictionary<string, object> { ["content"] = PROMPT };
            var json = JsonSerializer.Serialize(data);

            var reply = embeddingFunc(json);
            var replyData = JsonSerializer.Deserialize<Dictionary<string, object>>(reply);

            Assert.IsTrue(replyData.ContainsKey("embedding"));
            var embedding = (JsonElement)replyData["embedding"];

            int embedding_size = embeddingSizeFunc();
            Assert.AreEqual(embedding_size, embedding.GetArrayLength());
        }

        public void TestLoraList(Func<string> loraListFunc)
        {
            Console.WriteLine("LLM_Lora_List");
            var reply = loraListFunc();
            var replyData = JsonSerializer.Deserialize<JsonElement[]>(reply);
            Assert.AreEqual(0, replyData.Length);
        }

        public void TestCancel(Action<int> cancelAction)
        {
            Console.WriteLine("LLM_Cancel");
            cancelAction(ID_SLOT);
        }

        public void TestSlotSaveRestore(Func<string, string> slotFunc)
        {
            Console.WriteLine("LLM_Slot Save");

            string filename = "test_undreamai.save";
            string filepath = Path.Combine(Directory.GetCurrentDirectory(), filename);

            var saveData = new Dictionary<string, object>
            {
                ["id_slot"] = ID_SLOT,
                ["action"] = "save",
                ["filepath"] = filepath
            };

            var json = JsonSerializer.Serialize(saveData);
            var reply = slotFunc(json);
            var replyData = JsonSerializer.Deserialize<Dictionary<string, object>>(reply);
            Assert.AreEqual(filename, replyData["filename"].ToString());
            var nSaved = ((JsonElement)replyData["n_saved"]).GetInt32();
            Assert.IsTrue(nSaved > 0);
            Assert.IsTrue(File.Exists(filepath));

            Console.WriteLine("LLM_Slot Restore");

            var restoreData = new Dictionary<string, object>
            {
                ["id_slot"] = ID_SLOT,
                ["action"] = "restore",
                ["filepath"] = filepath
            };

            json = JsonSerializer.Serialize(restoreData);
            reply = slotFunc(json);
            replyData = JsonSerializer.Deserialize<Dictionary<string, object>>(reply);
            Assert.AreEqual(filename, replyData["filename"].ToString());
            var nRestored = ((JsonElement)replyData["n_restored"]).GetInt32();
            Assert.AreEqual(nSaved, nRestored);

            File.Delete(filepath);
        }


        [TestMethod]
        public void Tests_LlamaLib()
        {
            LlamaLib llamaLib = new LlamaLib();
            llamaLib.LLM_Debug(3);
            IntPtr llmService = llamaLib.LLMService_Construct(testModelPath);

            TestStart(() => llamaLib.LLM_Start(llmService), () => llamaLib.LLM_Started(llmService));
            TestTokenization(
                json => Marshal.PtrToStringAnsi(llamaLib.LLM_Tokenize(llmService, json)),
                json => Marshal.PtrToStringAnsi(llamaLib.LLM_Detokenize(llmService, json))
            );
            TestCompletion((json, cb) => Marshal.PtrToStringAnsi(llamaLib.LLM_Completion(llmService, json, cb)));
            TestEmbedding(json => Marshal.PtrToStringAnsi(llamaLib.LLM_Embeddings(llmService, json)), () => llamaLib.LLM_Embedding_Size(llmService));
            TestCancel((id_slot) => llamaLib.LLM_Cancel(llmService, id_slot));
            TestSlotSaveRestore(json => Marshal.PtrToStringAnsi(llamaLib.LLM_Slot(llmService, json)));
            TestLoraList(() => Marshal.PtrToStringAnsi(llamaLib.LLM_Lora_List(llmService)));

            Console.WriteLine("LLM_Cancel");
            llamaLib.LLM_Stop(llmService);
            Console.WriteLine("LLM_Delete");
            llamaLib.LLM_Delete(llmService);

            llamaLib?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMService()
        {
            LLMService llmService = new LLMService(testModelPath);
            llmService.Debug(3);

            TestStart(llmService.Start, llmService.Started);
            TestTokenization(llmService.Tokenize, llmService.Detokenize);
            TestCompletion(llmService.Completion);
            TestEmbedding(llmService.Embeddings, llmService.Embedding_Size);
            TestCancel(llmService.Cancel);
            TestSlotSaveRestore(llmService.Slot);
            TestLoraList(llmService.Lora_List);
            
            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMClient()
        {
            LLMService llmService = new LLMService(testModelPath);
            llmService.Debug(3);
            TestStart(llmService.Start, llmService.Started);

            LLMClient llmClient = new LLMClient(llmService);

            TestTokenization(llmClient.Tokenize, llmClient.Detokenize);
            TestCompletion(llmClient.Completion);
            TestEmbedding(llmClient.Embeddings, llmService.Embedding_Size);
            TestCancel(llmClient.Cancel);
            TestSlotSaveRestore(llmClient.Slot);

            llmService?.Dispose();
        }

        [TestMethod]
        public void Tests_LLMRemoteClient()
        {
            LLMService llmService = new LLMService(testModelPath);
            llmService.Debug(3);
            TestStart(llmService.Start, llmService.Started);
            llmService.Start_Server("", 13333);

            LLMRemoteClient llmClient = new LLMRemoteClient("http://localhost", 13333);

            TestTokenization(llmClient.Tokenize, llmClient.Detokenize);
            TestCompletion(llmClient.Completion);
            TestEmbedding(llmClient.Embeddings, llmService.Embedding_Size);

            llmService?.Dispose();
        }
    }
}