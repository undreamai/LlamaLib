using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
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

        public static void Debug(int debugLevel)
        {
            LlamaLib.Debug(debugLevel);
        }

        public static void LoggingCallback(LlamaLib.CharArrayCallback callback)
        {
            LlamaLib.LoggingCallback(callback);
        }

        public static void LoggingStop()
        {
            LlamaLib.LoggingStop();
        }

        protected void CheckLlamaLib()
        {
            if (disposed) throw new ObjectDisposedException(GetType().Name);
            if (llamaLib == null) throw new InvalidOperationException("LlamaLib instance is not initialized");
            if (llm == IntPtr.Zero) throw new InvalidOperationException("LLM instance is not initialized");
            if (llamaLib.LLM_Status_Code() != 0)
            {
                string status_msg = Marshal.PtrToStringAnsi(llamaLib.LLM_Status_Message()) ?? string.Empty;
                throw new AccessViolationException(status_msg);
            }
        }

        public virtual void Dispose() { }

        ~LLM()
        {
            Dispose();
        }

        public string GetTemplate()
        {
            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Get_Template(llm);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string ApplyTemplate(JArray messages = null)
        {
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));
            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Apply_Template(llm, messages.ToString());
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public List<int> Tokenize(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));

            CheckLlamaLib();
            IntPtr result = llamaLib.LLM_Tokenize(llm, content);
            string resultStr = Marshal.PtrToStringAnsi(result) ?? string.Empty;
            List<int> ret = new List<int>();
            try
            {
                JArray json = JArray.Parse(resultStr);
                ret = json?.ToObject<List<int>>();
            }
            catch { }
            return ret;
        }

        public string Detokenize(List<int> tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));

            CheckLlamaLib();
            JArray tokensJSON = JArray.FromObject(tokens);
            IntPtr result = llamaLib.LLM_Detokenize(llm, tokensJSON.ToString());
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Detokenize(int[] tokens)
        {
            if (tokens == null)
                throw new ArgumentNullException(nameof(tokens));
            return Detokenize(new List<int>(tokens));
        }

        public List<float> Embeddings(string content)
        {
            if (string.IsNullOrEmpty(content))
                throw new ArgumentNullException(nameof(content));

            CheckLlamaLib();

            IntPtr result = llamaLib.LLM_Embeddings(llm, content);
            string resultStr = Marshal.PtrToStringAnsi(result) ?? string.Empty;
            
            List<float> ret = new List<float>();
            try
            {
                JArray json = JArray.Parse(resultStr);
                ret = json?.ToObject<List<float>>();
            }
            catch { }
            return ret;
        }
     
        public string BuildParametersJSON(JObject parameters = null)
        {
            var json = new JObject
            {
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

        public void CheckCompletionInternal(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
                throw new ArgumentNullException(nameof(prompt));
            CheckLlamaLib();
        }

        public string CompletionInternal(string prompt, LlamaLib.CharArrayCallback callback, int idSlot, JObject parameters, bool callbackWithJSON)
        {
            IntPtr result;
            if (callbackWithJSON) result = llamaLib.LLM_Completion_JSON(llm, prompt, callback, idSlot, BuildParametersJSON(parameters));
            else result = llamaLib.LLM_Completion(llm, prompt, callback, idSlot, BuildParametersJSON(parameters));
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Completion(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1, JObject parameters = null)
        {
            CheckCompletionInternal(prompt);
            return CompletionInternal(prompt, callback, idSlot, parameters, false);
        }

        public async Task<string> CompletionAsync(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1, JObject parameters = null)
        {
            CheckCompletionInternal(prompt);
            return await Task.Run(() => CompletionInternal(prompt, callback, idSlot, parameters, false));
        }

        public JObject CompletionJSON(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1, JObject parameters = null)
        {
            CheckCompletionInternal(prompt);
            return JObject.Parse(CompletionInternal(prompt, callback, idSlot, parameters, true));
        }

        public async Task<JObject> CompletionJSONAsync(string prompt, LlamaLib.CharArrayCallback callback = null, int idSlot = -1, JObject parameters = null)
        {
            CheckCompletionInternal(prompt);
            return await Task.Run(() => JObject.Parse(CompletionInternal(prompt, callback, idSlot, parameters, true)));
        }
    }

    // LLMLocal class
    public abstract class LLMLocal : LLM
    {
        protected LLMLocal() : base() { }

        protected LLMLocal(LlamaLib llamaLibInstance) : base(llamaLibInstance) { }

        public string Slot(int idSlot, string action, string filepath)
        {
            if (string.IsNullOrEmpty(action))
                throw new ArgumentNullException(nameof(action));
            if (string.IsNullOrEmpty(filepath))
                throw new ArgumentNullException(nameof(filepath));

            IntPtr result = llamaLib.LLM_Slot(llm, idSlot, action, filepath);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
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


        public void SetTemplate(string template)
        {
            if (string.IsNullOrEmpty(template))
                throw new ArgumentNullException(nameof(template));
            CheckLlamaLib();
            llamaLib.LLM_Set_Template(llm, template);
        }

        // LoRA Weight methods
        public string BuildLoraWeightJSON(List<LoraIdScale> loras)
        {
            var jsonArray = new JArray();
            foreach (var lora in loras)
            {
                jsonArray.Add(new JObject { ["id"] = lora.Id, ["scale"] = lora.Scale });
            }
            return jsonArray.ToString();
        }

        public bool LoraWeight(List<LoraIdScale> loras)
        {
            if (loras == null)
                throw new ArgumentNullException(nameof(loras));
            
            var lorasJSON = BuildLoraWeightJSON(loras);
            return llamaLib.LLM_Lora_Weight(llm, lorasJSON);
        }

        public bool LoraWeight(params LoraIdScale[] loras)
        {
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
        public bool Start()
        {
            CheckLlamaLib();
            llamaLib.LLM_Start(llm);
            return llamaLib.LLM_Started(llm);
        }

        public async Task<bool> StartAsync()
        {
            CheckLlamaLib();
            return await Task.Run(() => {
                llamaLib.LLM_Start(llm);
                return llamaLib.LLM_Started(llm);
            });
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
                            llamaLib.LLM_Stop_Server(llm);
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

        public LLMClient(string url, int port)
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

        private IntPtr CreateClient(LLMProvider provider)
        {
            var llm = llamaLib.LLMClient_Construct(provider.llm);
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create LLMClient");
            return llm;
        }

        private IntPtr CreateRemoteClient(string url, int port)
        {
            var llm = llamaLib.LLMClient_Construct_Remote(url, port);
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create remote LLMClient for {url}:{port}");
            return llm;
        }
    }
}