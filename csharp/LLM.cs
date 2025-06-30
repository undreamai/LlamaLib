using System;
using System.IO;
using System.Runtime.InteropServices;

namespace UndreamAI.LlamaLib
{
    // Base LLM class
    public abstract class LLM : IDisposable
    {
        public LlamaLib llamaLib = null;
        public IntPtr llm = IntPtr.Zero;
        protected readonly object _disposeLock = new object();
        public bool disposed = false;

        protected LLM() { }

        protected LLM(LlamaLib llamaLibInstance)
        {
            llamaLib = llamaLibInstance ?? throw new ArgumentNullException(nameof(llamaLibInstance));
        }

        public void Debug(int level)
        {
            CheckLlamaLib();
            llamaLib.LLM_Debug(level);
        }

        public string Tokenize(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Tokenize(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Detokenize(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Detokenize(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Embeddings(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Embeddings(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string Completion(string jsonData, LlamaLib.CharArrayCallback callback = null)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Completion(llm, jsonData, callback);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        protected void CheckLlamaLib()
        {
            if (disposed)
                throw new ObjectDisposedException(GetType().Name);

            if (llamaLib == null)
                throw new InvalidOperationException("LlamaLib instance is not initialized");
            
            if (llm == IntPtr.Zero)
                throw new InvalidOperationException("LLM instance is not initialized");
        }

        public virtual void Dispose()
        {
        }

        ~LLM()
        {
            Dispose();
        }
    }

    // LLMLocal class
    public abstract class LLMLocal : LLM
    {
        protected LLMLocal() : base()
        {
        }

        protected LLMLocal(LlamaLib llamaLibInstance) : base(llamaLibInstance)
        {
        }

        public string Slot(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Slot(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public void Cancel(int idSlot)
        {
            CheckLlamaLib();
            
            llamaLib.LLM_Cancel(llm, idSlot);
        }
    }

    // LLMProvider class
    public abstract class LLMProvider : LLMLocal
    {
        protected LLMProvider() : base()
        {
        }

        protected LLMProvider(LlamaLib llamaLibInstance) : base(llamaLibInstance)
        {
        }

        public string LoraWeight(string jsonData)
        {
            if (string.IsNullOrEmpty(jsonData))
                throw new ArgumentNullException(nameof(jsonData));
            
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Lora_Weight(llm, jsonData);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

        public string LoraList()
        {
            CheckLlamaLib();
            
            var result = llamaLib.LLM_Lora_List(llm);
            return Marshal.PtrToStringAnsi(result) ?? string.Empty;
        }

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