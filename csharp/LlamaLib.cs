using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Collections.Generic;

namespace UndreamAI.LlamaLib
{
    public class LlamaLib
    {
        public string architecture { get; private set; }

        // Function delegates
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CharArrayCallback([MarshalAs(UnmanagedType.LPStr)] string charArray);

#if ANDROID || IOS || VISIONOS
        // Static P/Invoke declarations for mobile platforms
#if ANDROID_ARM64
        public const string DllName = "libllamalib_android-arm64";
#elif ANDROID_X64
        public const string DllName = "libllamalib_android-x64";
#else
        public const string DllName = "__Internal";
#endif

        public LlamaLib(bool gpu=false) {
#if ANDROID_ARM64
            architecture = "android-arm64";
#elif ANDROID_X64
            architecture = "android-x64";
#elif IOS
            architecture = "ios-arm64";
#elif VISIONOS
            architecture = "visionos-arm64";
#endif
        }

        // Base LLM functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Debug")]
        public static extern void LLM_Debug_Static(int debugLevel);
        public static void Debug(int debugLevel) => LLM_Debug_Static(debugLevel);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Logging_Callback")]
        public static extern void LLM_Logging_Callback_Static(CharArrayCallback callback);
        public static void LoggingCallback(CharArrayCallback callback) => LLM_Logging_Callback_Static(callback);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Logging_Stop")]
        public static extern void LLM_Logging_Stop_Static();
        public static void LoggingStop() => LLM_Logging_Stop_Static();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Tokenize")]
        public static extern IntPtr LLM_Tokenize_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        public IntPtr LLM_Tokenize(IntPtr llm, string jsonData) => LlamaLib.LLM_Tokenize_Static(llm, jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Detokenize")]
        public static extern IntPtr LLM_Detokenize_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        public IntPtr LLM_Detokenize(IntPtr llm, string jsonData) => LlamaLib.LLM_Detokenize_Static(llm, jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Embeddings")]
        public static extern IntPtr LLM_Embeddings_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        public IntPtr LLM_Embeddings(IntPtr llm, string jsonData) => LlamaLib.LLM_Embeddings_Static(llm, jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Completion")]
        public static extern IntPtr LLM_Completion_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData, CharArrayCallback callback, bool callbackWithJSON=true);
        public IntPtr LLM_Completion(IntPtr llm, string jsonData, CharArrayCallback callback, bool callbackWithJSON) => LlamaLib.LLM_Completion_Static(llm, jsonData, callback, callbackWithJSON);

        // LLMLocal functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Slot")]
        public static extern IntPtr LLM_Slot_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        public IntPtr LLM_Slot(IntPtr llm, string jsonData) => LlamaLib.LLM_Slot_Static(llm, jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Cancel")]
        public static extern void LLM_Cancel_Static(IntPtr llm, int idSlot);
        public void LLM_Cancel(IntPtr llm, int idSlot) => LlamaLib.LLM_Cancel_Static(llm, idSlot);

        // LLMProvider functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Lora_Weight")]
        public static extern IntPtr LLM_Lora_Weight_Static(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        public IntPtr LLM_Lora_Weight(IntPtr llm, string jsonData) => LlamaLib.LLM_Lora_Weight_Static(llm, jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Lora_List")]
        public static extern IntPtr LLM_Lora_List_Static(IntPtr llm);
        public IntPtr LLM_Lora_List(IntPtr llm) => LlamaLib.LLM_Lora_List_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Delete")]
        public static extern void LLM_Delete_Static(IntPtr llm);
        public void LLM_Delete(IntPtr llm) => LlamaLib.LLM_Delete_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Start")]
        public static extern void LLM_Start_Static(IntPtr llm);
        public void LLM_Start(IntPtr llm) => LlamaLib.LLM_Start_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Started")]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool LLM_Started_Static(IntPtr llm);
        public bool LLM_Started(IntPtr llm) => LlamaLib.LLM_Started_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Stop")]
        public static extern void LLM_Stop_Static(IntPtr llm);
        public void LLM_Stop(IntPtr llm) => LlamaLib.LLM_Stop_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Start_Server")]
        public static extern void LLM_Start_Server_Static(IntPtr llm, 
            [MarshalAs(UnmanagedType.LPStr)] string host="0.0.0.0", 
            int port=0, 
            [MarshalAs(UnmanagedType.LPStr)] string apiKey="");
        public void LLM_Start_Server(IntPtr llm, string host="0.0.0.0", int port=0, string apiKey="") => LlamaLib.LLM_Start_Server_Static(llm, host, port, apiKey);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Stop_Server")]
        public static extern void LLM_Stop_Server_Static(IntPtr llm);
        public void LLM_Stop_Server(IntPtr llm) => LlamaLib.LLM_Stop_Server_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Join_Service")]
        public static extern void LLM_Join_Service_Static(IntPtr llm);
        public void LLM_Join_Service(IntPtr llm) => LlamaLib.LLM_Join_Service_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Join_Server")]
        public static extern void LLM_Join_Server_Static(IntPtr llm);
        public void LLM_Join_Server(IntPtr llm) => LlamaLib.LLM_Join_Server_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Set_SSL")]
        public static extern void LLM_Set_SSL_Static(IntPtr llm, 
            [MarshalAs(UnmanagedType.LPStr)] string sslCert, 
            [MarshalAs(UnmanagedType.LPStr)] string sslKey);
        public void LLM_Set_SSL(IntPtr llm, string sslCert, string sslKey) => LlamaLib.LLM_Set_SSL_Static(llm, sslCert, sslKey);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Status_Code")]
        public static extern int LLM_Status_Code_Static(IntPtr llm);
        public int LLM_Status_Code(IntPtr llm) => LlamaLib.LLM_Status_Code_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Status_Message")]
        public static extern IntPtr LLM_Status_Message_Static(IntPtr llm);
        public IntPtr LLM_Status_Message(IntPtr llm) => LlamaLib.LLM_Status_Message_Static(llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Embedding_Size")]
        public static extern int LLM_Embedding_Size_Static(IntPtr llm);
        public int LLM_Embedding_Size(IntPtr llm) => LlamaLib.LLM_Embedding_Size_Static(llm);

        // LLMService functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMService_Construct")]
        public static extern IntPtr LLMService_Construct_Static(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads = -1,
            int numGpuLayers = 0,
            int numParallel = 1,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention = false,
            int contextSize = 4096,
            int batchSize = 2048,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly = false,
            int loraCount = 0,
            IntPtr loraPaths = default);
        public IntPtr LLMService_Construct(
            string modelPath,
            int numThreads = -1,
            int numGpuLayers = 0,
            int numParallel = 1,
            bool flashAttention = false,
            int contextSize = 4096,
            int batchSize = 2048,
            bool embeddingOnly = false,
            int loraCount = 0,
            IntPtr loraPaths = default)
            => LlamaLib.LLMService_Construct_Static(modelPath, numThreads, numGpuLayers, numParallel, flashAttention,
                                            contextSize, batchSize, embeddingOnly, loraCount, loraPaths);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMService_From_Command")]
        public static extern IntPtr LLMService_From_Command_Static([MarshalAs(UnmanagedType.LPStr)] string paramsString);
        public IntPtr LLMService_From_Command(string paramsString) => LlamaLib.LLMService_From_Command_Static(paramsString);

        // LLMClient functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMClient_Construct")]
        public static extern IntPtr LLMClient_Construct_Static(IntPtr llm);
        public IntPtr LLMClient_Construct(IntPtr llm) => LlamaLib.LLMClient_Construct_Static(llm);

        // LLMRemoteClient functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMRemoteClient_Construct")]
        public static extern IntPtr LLMRemoteClient_Construct_Static(
            [MarshalAs(UnmanagedType.LPStr)] string url,
            int port);
        public IntPtr LLMRemoteClient_Construct(string url, int port) => LlamaLib.LLMRemoteClient_Construct_Static(url, port);
#else
        // Desktop platform implementation with dynamic loading
        private static List<LlamaLib> instances = new List<LlamaLib>();
        private static readonly object runtimeLock = new object();
        private static IntPtr runtimeLibraryHandle = IntPtr.Zero;
        private IntPtr libraryHandle = IntPtr.Zero;
        private static int debugLevelGlobal = 0;
        private static CharArrayCallback loggingCallbackGlobal = null;

        static LlamaLib()
        {
            LoadRuntimeLibrary();
        }

        private static void LoadRuntimeLibrary()
        {
            lock (runtimeLock)
            {
                if (runtimeLibraryHandle == IntPtr.Zero)
                {
                    runtimeLibraryHandle = LibraryLoader.LoadLibrary(GetRuntimeLibraryPath());
                    Has_GPU_Layers = LibraryLoader.GetSymbolDelegate<Has_GPU_Layers_Delegate>(runtimeLibraryHandle, "Has_GPU_Layers");
                    Available_Architectures = LibraryLoader.GetSymbolDelegate<Available_Architectures_Delegate>(runtimeLibraryHandle, "Available_Architectures");
                }
            }
        }

        public LlamaLib(bool gpu = false)
        {
            LoadLibraries(gpu);
            lock (runtimeLock)
            {
                instances.Add(this);
                LLM_Debug(debugLevelGlobal);
                if(loggingCallbackGlobal != null) LLM_Logging_Callback(loggingCallbackGlobal);
            }
        }

        public static string GetPlatform()
        {

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return "linux-x64";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                if (RuntimeInformation.ProcessArchitecture == Architecture.X64)
                    return "osx-x64";
                else
                    return "osx-arm64";
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "win-x64";
            else throw new ArgumentException("Unknown platform " + RuntimeInformation.OSDescription);
        }

        public static string FindLibrary(string libraryName)
        {
            string baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);

            List<string> lookupDirs = new List<string>();
            lookupDirs.Add(Path.Combine(baseDir, "runtimes", GetPlatform(), "native"));
            lookupDirs.Add(baseDir);

            foreach (string lookupDir in lookupDirs)
            {
                string libraryPath = Path.Combine(lookupDir, libraryName);
                if (File.Exists(libraryPath)) return libraryPath;
            }

            throw new InvalidOperationException($"Library {libraryName} not found!");
        }

        static string GetRuntimeLibraryPath()
        {
            string platform = GetPlatform();
            string libName;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                libName = "libllamalib_" + platform + "_runtime.so";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                libName = "libllamalib_" + platform + "_runtime.dylib";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                libName = "llamalib_" + platform + "_runtime.dll";
            else
                throw new ArgumentException("Unknown platform " + RuntimeInformation.OSDescription);
            return FindLibrary(libName);
        }

        private void LoadLibraries(bool gpu)
        {
            string architecturesString = Marshal.PtrToStringAnsi(Available_Architectures(gpu));
            if (string.IsNullOrEmpty(architecturesString))
            {
                throw new InvalidOperationException("No architectures available for the specified GPU setting.");
            }

            string[] libraries = architecturesString.Split(',');
            Exception lastException = null;

            foreach (string library in libraries)
            {
                try
                {
                    string libraryPath = FindLibrary(library.Trim());
                    if (debugLevelGlobal > 0) Console.WriteLine("Trying " + libraryPath);
                    libraryHandle = LibraryLoader.LoadLibrary(libraryPath);
                    LoadFunctionPointers();
                    architecture = library.Trim();
                    if (debugLevelGlobal > 0) Console.WriteLine("Successfully loaded: " + libraryPath);
                    return;
                }
                catch (Exception ex)
                {
                    if (debugLevelGlobal > 0) Console.WriteLine($"Failed to load library {library}: {ex.Message}.");
                    lastException = ex;
                    continue;
                }
            }

            // If we get here, no library was successfully loaded
            throw new InvalidOperationException($"Failed to load any library. Available libraries: {string.Join(", ", libraries)}", lastException);
        }

        private void LoadFunctionPointers()
        {
            LLM_Debug = LibraryLoader.GetSymbolDelegate<LLM_Debug_Delegate>(libraryHandle, "LLM_Debug");
            LLM_Logging_Callback = LibraryLoader.GetSymbolDelegate<LLM_Logging_Callback_Delegate>(libraryHandle, "LLM_Logging_Callback");
            LLM_Logging_Stop = LibraryLoader.GetSymbolDelegate<LLM_Logging_Stop_Delegate>(libraryHandle, "LLM_Logging_Stop");
            LLM_Tokenize = LibraryLoader.GetSymbolDelegate<LLM_Tokenize_Delegate>(libraryHandle, "LLM_Tokenize");
            LLM_Detokenize = LibraryLoader.GetSymbolDelegate<LLM_Detokenize_Delegate>(libraryHandle, "LLM_Detokenize");
            LLM_Embeddings = LibraryLoader.GetSymbolDelegate<LLM_Embeddings_Delegate>(libraryHandle, "LLM_Embeddings");
            LLM_Completion = LibraryLoader.GetSymbolDelegate<LLM_Completion_Delegate>(libraryHandle, "LLM_Completion");
            LLM_Slot = LibraryLoader.GetSymbolDelegate<LLM_Slot_Delegate>(libraryHandle, "LLM_Slot");
            LLM_Cancel = LibraryLoader.GetSymbolDelegate<LLM_Cancel_Delegate>(libraryHandle, "LLM_Cancel");
            LLM_Lora_Weight = LibraryLoader.GetSymbolDelegate<LLM_Lora_Weight_Delegate>(libraryHandle, "LLM_Lora_Weight");
            LLM_Lora_List = LibraryLoader.GetSymbolDelegate<LLM_Lora_List_Delegate>(libraryHandle, "LLM_Lora_List");
            LLM_Delete = LibraryLoader.GetSymbolDelegate<LLM_Delete_Delegate>(libraryHandle, "LLM_Delete");
            LLM_Start = LibraryLoader.GetSymbolDelegate<LLM_Start_Delegate>(libraryHandle, "LLM_Start");
            LLM_Started = LibraryLoader.GetSymbolDelegate<LLM_Started_Delegate>(libraryHandle, "LLM_Started");
            LLM_Stop = LibraryLoader.GetSymbolDelegate<LLM_Stop_Delegate>(libraryHandle, "LLM_Stop");
            LLM_Start_Server = LibraryLoader.GetSymbolDelegate<LLM_Start_Server_Delegate>(libraryHandle, "LLM_Start_Server");
            LLM_Stop_Server = LibraryLoader.GetSymbolDelegate<LLM_Stop_Server_Delegate>(libraryHandle, "LLM_Stop_Server");
            LLM_Join_Service = LibraryLoader.GetSymbolDelegate<LLM_Join_Service_Delegate>(libraryHandle, "LLM_Join_Service");
            LLM_Join_Server = LibraryLoader.GetSymbolDelegate<LLM_Join_Server_Delegate>(libraryHandle, "LLM_Join_Server");
            LLM_Set_SSL = LibraryLoader.GetSymbolDelegate<LLM_Set_SSL_Delegate>(libraryHandle, "LLM_Set_SSL");
            LLM_Status_Code = LibraryLoader.GetSymbolDelegate<LLM_Status_Code_Delegate>(libraryHandle, "LLM_Status_Code");
            LLM_Status_Message = LibraryLoader.GetSymbolDelegate<LLM_Status_Message_Delegate>(libraryHandle, "LLM_Status_Message");
            LLM_Embedding_Size = LibraryLoader.GetSymbolDelegate<LLM_Embedding_Size_Delegate>(libraryHandle, "LLM_Embedding_Size");
            LLMService_Construct = LibraryLoader.GetSymbolDelegate<LLMService_Construct_Delegate>(libraryHandle, "LLMService_Construct");
            LLMService_From_Command = LibraryLoader.GetSymbolDelegate<LLMService_From_Command_Delegate>(libraryHandle, "LLMService_From_Command");
            LLMClient_Construct = LibraryLoader.GetSymbolDelegate<LLMClient_Construct_Delegate>(libraryHandle, "LLMClient_Construct");
            LLMRemoteClient_Construct = LibraryLoader.GetSymbolDelegate<LLMRemoteClient_Construct_Delegate>(libraryHandle, "LLMRemoteClient_Construct");
        }

        // Delegate definitions for desktop platforms
        // Runtime lib
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr Available_Architectures_Delegate([MarshalAs(UnmanagedType.I1)] bool gpu);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate bool Has_GPU_Layers_Delegate([MarshalAs(UnmanagedType.LPStr)] string command);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Debug_Delegate(int debugLevel);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Logging_Callback_Delegate(CharArrayCallback callback);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Logging_Stop_Delegate();

        // Main lib
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Tokenize_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Detokenize_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Embeddings_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Completion_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData, CharArrayCallback callback, bool callbackWithJSON=true);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Slot_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Cancel_Delegate(IntPtr llm, int idSlot);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Lora_Weight_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Lora_List_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Delete_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Start_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate bool LLM_Started_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Stop_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Start_Server_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string host="0.0.0.0", int port=0, [MarshalAs(UnmanagedType.LPStr)] string apiKey="");
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Stop_Server_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Join_Service_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Join_Server_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LLM_Set_SSL_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string sslCert, [MarshalAs(UnmanagedType.LPStr)] string sslKey);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int LLM_Status_Code_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Status_Message_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int LLM_Embedding_Size_Delegate(IntPtr llm);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMService_Construct_Delegate(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads = -1,
            int numGpuLayers = 0,
            int numParallel = 1,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention = false,
            int contextSize = 4096,
            int batchSize = 2048,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly = false,
            int loraCount = 0,
            IntPtr loraPaths = default);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMService_From_Command_Delegate([MarshalAs(UnmanagedType.LPStr)] string paramsString);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMClient_Construct_Delegate(IntPtr llm);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMRemoteClient_Construct_Delegate([MarshalAs(UnmanagedType.LPStr)] string url, int port);

        // Function pointers for desktop platforms
        // Runtime lib
        public static Available_Architectures_Delegate Available_Architectures;
        public static Has_GPU_Layers_Delegate Has_GPU_Layers;

        // Main lib
        public LLM_Debug_Delegate LLM_Debug;
        public LLM_Logging_Callback_Delegate LLM_Logging_Callback;
        public LLM_Logging_Stop_Delegate LLM_Logging_Stop;
        public LLM_Tokenize_Delegate LLM_Tokenize;
        public LLM_Detokenize_Delegate LLM_Detokenize;
        public LLM_Embeddings_Delegate LLM_Embeddings;
        public LLM_Completion_Delegate LLM_Completion;
        public LLM_Slot_Delegate LLM_Slot;
        public LLM_Cancel_Delegate LLM_Cancel;
        public LLM_Lora_Weight_Delegate LLM_Lora_Weight;
        public LLM_Lora_List_Delegate LLM_Lora_List;
        public LLM_Delete_Delegate LLM_Delete;
        public LLM_Start_Delegate LLM_Start;
        public LLM_Started_Delegate LLM_Started;
        public LLM_Stop_Delegate LLM_Stop;
        public LLM_Start_Server_Delegate LLM_Start_Server;
        public LLM_Stop_Server_Delegate LLM_Stop_Server;
        public LLM_Join_Service_Delegate LLM_Join_Service;
        public LLM_Join_Server_Delegate LLM_Join_Server;
        public LLM_Set_SSL_Delegate LLM_Set_SSL;
        public LLM_Status_Code_Delegate LLM_Status_Code;
        public LLM_Status_Message_Delegate LLM_Status_Message;
        public LLM_Embedding_Size_Delegate LLM_Embedding_Size;
        public LLMService_Construct_Delegate LLMService_Construct;
        public LLMService_From_Command_Delegate LLMService_From_Command;
        public LLMClient_Construct_Delegate LLMClient_Construct;
        public LLMRemoteClient_Construct_Delegate LLMRemoteClient_Construct;

        public static void Debug(int debugLevel)
        {
            debugLevelGlobal = debugLevel;
            foreach (LlamaLib instance in instances)
            {
                instance.LLM_Debug(debugLevel);
            }
        }

        public static void LoggingCallback(CharArrayCallback callback)
        {
            loggingCallbackGlobal = callback;
            foreach (LlamaLib instance in instances)
            {
                instance.LLM_Logging_Callback(callback);
            }
        }

        public static void LoggingStop()
        {
            LoggingCallback(null);
        }

        public void Dispose()
        {
            LibraryLoader.FreeLibrary(libraryHandle);
            libraryHandle = IntPtr.Zero;

            lock (runtimeLock)
            {
                instances.Remove(this);
                if (instances.Count == 0)
                {
                    LibraryLoader.FreeLibrary(runtimeLibraryHandle);
                    runtimeLibraryHandle = IntPtr.Zero;
                }
            }
        }

        ~LlamaLib()
        {
            Dispose();
        }
#endif
    }
}