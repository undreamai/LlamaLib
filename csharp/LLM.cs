using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace UndreamAI.LlamaLib
{

    // Data structures for LoRA operations
    public struct LoraIdScale
    {
        public int Id { get; set; }
        public float Scale { get; set; }
        
        public LoraIdScale(int id, float scale)
        {
            Id = id;
            Scale = scale;
        }
    }

    public struct LoraIdScalePath
    {
        public int Id { get; set; }
        public float Scale { get; set; }
        public string Path { get; set; }
        
        public LoraIdScalePath(int id, float scale, string path)
        {
            Id = id;
            Scale = scale;
            Path = path;
        }
    }

    // Base LLM class
    public abstract class LLM : IDisposable
    {
        public LlamaLib llamaLib = null;
        public IntPtr llm = IntPtr.Zero;
        protected readonly object _disposeLock = new object();
        public bool disposed = false;
        public int seed = 0;
        public int numPredict = -1;
        public int numKeep = 0;
        public float temperature = 0.80f;
        public string jsonSchema = "";
        public string grammar = "";

        protected LLM() { }

        protected LLM(LlamaLib llamaLibInstance)
        {
            llamaLib = llamaLibInstance ?? throw new ArgumentNullException(nameof(llamaLibInstance));
        }

        public void Debug(bool debug)
        {
            CheckLlamaLib();
            llamaLib.LLM_Debug(debug);
        }

        protected void CheckLlamaLib()
        {
            if (disposed) throw new ObjectDisposedException(GetType().Name);
            if (llamaLib == null) throw new InvalidOperationException("LlamaLib instance is not initialized");
            if (llm == IntPtr.Zero) throw new InvalidOperationException("LLM instance is not initialized");
        }

        public virtual void Dispose() { }

        ~LLM()
        {
            Dispose();
        }

        // Tokenize methods
        public string BuildTokenizeJSON(string content)
        {
            var json = new JObject{["content"] = content};
            return json.ToString();
        }

        public List<int> ParseTokenizeJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["tokens"]?.ToObject<List<int>>() ?? new List<int>();
            }
            catch
            {
                return new List<int>();
            }
        }
        
        public string TokenizeJSON(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Tokenize(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public List<int> Tokenize(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));
            
            var jsonInput = BuildTokenizeJSON(content);
            var jsonResult = TokenizeJSON(jsonInput);
            return ParseTokenizeJSON(jsonResult);
        }

        // Detokenize methods
        public string BuildDetokenizeJSON(List<int> tokens)
        {
            var json = new JObject{["tokens"] = JArray.FromObject(tokens)};
            return json.ToString();
        }

        public string ParseDetokenizeJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["content"]?.ToString() ?? string.Empty;
            }
            catch
            {
                return string.Empty;
            }
        }

        public string DetokenizeJSON(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));

            CheckLlamaLib();

            var result = llamaLib.LLM_Detokenize(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Detokenize(List<int> tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            
            var jsonInput = BuildDetokenizeJSON(tokens);
            var jsonResult = DetokenizeJSON(jsonInput);
            return ParseDetokenizeJSON(jsonResult);
        }

        public string Detokenize(int[] tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            
            return Detokenize(new List<int>(tokens));
        }

        // Embeddings methods
        public string BuildEmbeddingsJSON(string content)
        {
            var json = new JObject{["content"] = content};
            return json.ToString();
        }

        public List<float> ParseEmbeddingsJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["embedding"]?.ToObject<List<float>>() ?? new List<float>();
            }
            catch
            {
                return new List<float>();
            }
        }

        public string EmbeddingsJSON(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));

            CheckLlamaLib();

            var result = llamaLib.LLM_Embeddings(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public List<float> Embeddings(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));
            
            var jsonInput = BuildEmbeddingsJSON(content);
            var jsonResult = EmbeddingsJSON(jsonInput);
            return ParseEmbeddingsJSON(jsonResult);
        }

        // Completion methods        
        public string BuildCompletionJSON(string prompt, int idSlot = 0, JObject parameters = null)
        {
            var json = new JObject
            {
                ["prompt"] = prompt,
                ["id_slot"] = idSlot,
                ["seed"] = seed,
                ["n_predict"] = numPredict,
                ["n_keep"] = numKeep,
                ["temperature"] = temperature,
            };
            if (jsonSchema != "") json["json_schema"] = jsonSchema;
            if (grammar != "") json["grammar"] = grammar;
            if (parameters != null)
            {
                foreach (var param in parameters)
                {
                    json[param.Key] = param.Value;
                }
            }
            return json.ToString();
        }

        public static string ParseCompletionJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["content"]?.ToString() ?? string.Empty;
            }
            catch
            {
                return string.Empty;
            }
        }

        public string CompletionJSON(string jsonData, LlamaLib.CharArrayCallback callback = null)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));

            CheckLlamaLib();

            var result = llamaLib.LLM_Completion(llm, jsonData, callback, true);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Completion(string prompt, int idSlot = -1, LlamaLib.CharArrayCallback callback = null, JObject parameters = null)
        {
            if (string.IsNullOrEmpty(prompt))
                throw new ArgumentNullException(nameof(prompt));

            CheckLlamaLib();

            var jsonData = BuildCompletionJSON(prompt, idSlot, parameters);
            var result = llamaLib.LLM_Completion(llm, jsonData, callback, false);
            var jsonResult = Marshal.PtrToStringAnsi(result) ?? string.Empty;
            return ParseCompletionJSON(jsonResult);
        }
    }

    // LLMLocal class
    public abstract class LLMLocal : LLM
    {
        protected LLMLocal() : base() { }

        protected LLMLocal(LlamaLib llamaLibInstance) : base(llamaLibInstance) { }

        // Slot methods
        public string BuildSlotJSON(int idSlot, string action, string filepath)
        {
            var json = new JObject{["id_slot"] = idSlot, ["action"] = action, ["filepath"] = filepath};
            return json.ToString();
        }

        public string ParseSlotJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["filename"]?.ToString() ?? string.Empty;
            }
            catch
            {
                return string.Empty;
            }
        }

        public string SlotJSON(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));

            CheckLlamaLib();

            var result = llamaLib.LLM_Slot(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Slot(int idSlot, string action, string filepath)
        {
            if (string.IsNullOrEmpty(action))
                throw new ArgumentNullException(nameof(action));
            if (string.IsNullOrEmpty(filepath))
                throw new ArgumentNullException(nameof(filepath));
            
            var jsonInput = BuildSlotJSON(idSlot, action, filepath);
            var jsonResult = SlotJSON(jsonInput);
            return ParseSlotJSON(jsonResult);
        }

        // Cancel methods
        public void Cancel(int idSlot)
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Cancel(llm, idSlot);
        }
    }

    // LLMProvider class
    public abstract class LLMProvider : LLMLocal
    {
        protected LLMProvider() : base() { }

        protected LLMProvider(LlamaLib llamaLibInstance) : base(llamaLibInstance) { }

        // LoRA Weight methods
        public string BuildLoraWeightJSON(List<LoraIdScale> loras)
        {
            var jsonArray = new JArray();
            foreach (var lora in loras)
            {
                jsonArray.Add(new JObject{["id"] = lora.Id, ["scale"] = lora.Scale});
            }
            return jsonArray.ToString();
        }

        public bool ParseLoraWeightJSON(string result)
        {
            try
            {
                var json = JObject.Parse(result);
                return json["success"]?.ToObject<bool>() ?? false;
            }
            catch
            {
                return false;
            }
        }
        
        public string LoraWeightJSON(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Lora_Weight(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public bool LoraWeight(List<LoraIdScale> loras)
        {
            if (loras == null)
                throw new ArgumentNullException(nameof(loras));
            
            var jsonInput = BuildLoraWeightJSON(loras);
            var jsonResult = LoraWeightJSON(jsonInput);
            return ParseLoraWeightJSON(jsonResult);
        }

        public bool LoraWeight(params LoraIdScale[] loras)
        {
            if (loras == null)
                throw new ArgumentNullException(nameof(loras));

            return LoraWeight(new List<LoraIdScale>(loras));
        }

        // LoRA List methods
        public List<LoraIdScalePath> ParseLoraListJSON(string result)
        {
            var loras = new List<LoraIdScalePath>();
            try
            {
                var jsonArray = JArray.Parse(result);
                foreach (var item in jsonArray)
                {
                    int id = item["id"]?.ToObject<int>() ?? -1;
                    if (id < 0) continue;
                    loras.Add(new LoraIdScalePath(
                        id,
                        item["scale"]?.ToObject<float>() ?? 0.0f,
                        item["path"]?.ToString() ?? string.Empty
                    ));
                }
            }
            catch{ }
            return loras;
        }

        public string LoraListJSON()
        {
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Lora_List(llm);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public List<LoraIdScalePath> LoraList()
        {
            var jsonResult = LoraListJSON();
            return ParseLoraListJSON(jsonResult);
        }

        // Server methods
        public void Start()
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Start(llm);
        }

        public bool Started()
        {
            CheckLlamaLib();
            
            return llamaLib.LLM_Started(llm);
        }

        public void Stop()
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Stop(llm);
        }

        public void StartServer(string host = "0.0.0.0", int port = 0, string apiKey = "")
        {
            CheckLlamaLib();
            
            if (string.IsNullOrEmpty(host))
                host = "0.0.0.0";
            
            llamaLib.LLM_Start_Server(llm, host, port, apiKey ?? string.Empty);
        }

        public void StopServer()
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Stop_Server(llm);
        }

        public void JoinService()
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Join_Service(llm);
        }

        public void JoinServer()
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Join_Server(llm);
        }

        public void SetSSL(string sslCert, string sslKey)
        {
            if (string.IsNullOrEmpty(sslCert))
                throw new ArgumentNullException(nameof(sslCert));
            if (string.IsNullOrEmpty(sslKey))
                throw new ArgumentNullException(nameof(sslKey));
            
            CheckLlamaLib();
            
            llamaLib.LLM_Set_SSL(llm, sslCert, sslKey);
        }

        public int StatusCode()
        {
            CheckLlamaLib();
            return llamaLib.LLM_Status_Code(llm);
        }

        public string StatusMessage()
        {
            CheckLlamaLib();
            var result = llamaLib.LLM_Status_Message(llm);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public int EmbeddingSize()
        {
            CheckLlamaLib();
            return llamaLib.LLM_Embedding_Size(llm);
        }

        public override void Dispose()
        {
            lock (_disposeLock)
            {
                if (!disposed)
                {
                    if (llm != IntPtr.Zero && llamaLib != null)
                    {
                        try
                        {
                            llamaLib.LLM_Stop(llm);
                            llamaLib.LLM_Delete(llm);
                        }
                        catch (Exception){ }
                    }
                    llamaLib?.Dispose();
                    llamaLib = null;
                    llm = IntPtr.Zero;
                }
                disposed = true;
            }
        }
    }

    // LLMService class
    public class LLMService : LLMProvider
    {
        public LLMService(string modelPath, int numThreads = -1, int numGpuLayers = 0, 
            int numParallel = 1, bool flashAttention = false, int contextSize = 4096, 
            int batchSize = 2048, bool embeddingOnly = false, string[] loraPaths = null)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentNullException(nameof(modelPath));
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found: {modelPath}");

            try
            {
                llamaLib = new LlamaLib(numGpuLayers > 0);
                llm = CreateLLM(modelPath, numThreads, numGpuLayers, numParallel, 
                    flashAttention, contextSize, batchSize, embeddingOnly, loraPaths);
            }
            catch
            {
                llamaLib?.Dispose();
                throw;
            }
        }

        public LLMService(LlamaLib llamaLibInstance, IntPtr llmInstance)
        {
            if (llamaLibInstance == null) throw new ArgumentNullException(nameof(llamaLibInstance));
            if (llmInstance == IntPtr.Zero) throw new ArgumentNullException(nameof(llmInstance));
            llamaLib = llamaLibInstance;
            llm = llmInstance;
        }

        public static LLMService FromCommand(string paramsString)
        {
            if (string.IsNullOrEmpty(paramsString))
                throw new ArgumentNullException(nameof(paramsString));

            LlamaLib llamaLibInstance = null;
            IntPtr llmInstance = IntPtr.Zero;
            try
            {
                llamaLibInstance = new LlamaLib(LlamaLib.Has_GPU_Layers(paramsString));
                llmInstance = llamaLibInstance.LLMService_From_Command(paramsString);
            }
            catch
            {
                llamaLibInstance?.Dispose();
                throw;
            }
            return new LLMService(llamaLibInstance, llmInstance);
        }

        private IntPtr CreateLLM(string modelPath, int numThreads, int numGpuLayers,
            int numParallel, bool flashAttention, int contextSize, int batchSize,
            bool embeddingOnly, string[] loraPaths)
        {
            IntPtr loraPathsPtr = IntPtr.Zero;
            int loraPathCount = 0;

            if (loraPaths != null && loraPaths.Length > 0)
            {
                loraPathCount = loraPaths.Length;
                // Allocate array of string pointers
                loraPathsPtr = Marshal.AllocHGlobal(IntPtr.Size * loraPathCount);
                
                try
                {
                    for (int i = 0; i < loraPathCount; i++)
                    {
                        if (string.IsNullOrEmpty(loraPaths[i]))
                            throw new ArgumentException($"Lora path at index {i} is null or empty");
                        
                        IntPtr stringPtr = Marshal.StringToHGlobalAnsi(loraPaths[i]);
                        Marshal.WriteIntPtr(loraPathsPtr, i * IntPtr.Size, stringPtr);
                    }
                }
                catch
                {
                    // Clean up if allocation failed
                    for (int i = 0; i < loraPathCount; i++)
                    {
                        IntPtr stringPtr = Marshal.ReadIntPtr(loraPathsPtr, i * IntPtr.Size);
                        if (stringPtr != IntPtr.Zero)
                            Marshal.FreeHGlobal(stringPtr);
                    }
                    Marshal.FreeHGlobal(loraPathsPtr);
                    throw;
                }
            }

            try
            {
                var llm = llamaLib.LLMService_Construct(
                    modelPath, numThreads, numGpuLayers, numParallel,
                    flashAttention, contextSize, batchSize, embeddingOnly,
                    loraPathCount, loraPathsPtr);

                if (llm == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to create LLMService");

                return llm;
            }
            finally
            {
                // Clean up allocated strings
                if (loraPathsPtr != IntPtr.Zero)
                {
                    for (int i = 0; i < loraPathCount; i++)
                    {
                        IntPtr stringPtr = Marshal.ReadIntPtr(loraPathsPtr, i * IntPtr.Size);
                        if (stringPtr != IntPtr.Zero)
                            Marshal.FreeHGlobal(stringPtr);
                    }
                    Marshal.FreeHGlobal(loraPathsPtr);
                }
            }
        }
    }

    // LLMClient class
    public class LLMClient : LLMLocal
    {
        public LLMClient(LLMProvider provider)
        {
            if (provider.disposed)
                throw new ObjectDisposedException(nameof(provider));

            llamaLib = provider.llamaLib;
            llm = CreateClient(provider);
        }

        private IntPtr CreateClient(LLMProvider provider)
        {
            var llm = llamaLib.LLMClient_Construct(provider.llm);
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create LLMClient");
            return llm;
        }
    }

    // LLMRemoteClient class
    public class LLMRemoteClient : LLM
    {
        public LLMRemoteClient(string url, int port)
        {
            if (string.IsNullOrEmpty(url))
                throw new ArgumentNullException(nameof(url));

            try
            {
                llamaLib = new LlamaLib(false);
                llm = CreateRemoteClient(url, port);
            }
            catch
            {
                llamaLib?.Dispose();
                throw;
            }
        }

        private IntPtr CreateRemoteClient(string url, int port)
        {
            var llm = llamaLib.LLMRemoteClient_Construct(url, port);
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create LLMRemoteClient for {url}:{port}");
            return llm;
        }
    }
}