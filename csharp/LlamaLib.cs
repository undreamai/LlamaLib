using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Reflection;

namespace UndreamAI.LlamaLib
{
    public class LlamaLib
    {
        public string architecture { get; private set; }
        private IntPtr libraryHandle = IntPtr.Zero;

        private static IntPtr runtimeLibraryHandle = IntPtr.Zero;
        private static string libraryBasePath => GetLibraryBasePath();

        // Mobile platforms (static linking)
#if ANDROID
        public const string DllName = "libllamalib_android";
#elif IOS
        public const string DllName = "__Internal";
#elif VISIONOS
        public const string DllName = "__Internal";
#else
        // Desktop platforms use dynamic loading
        private const string DllName = null;
#endif

        // Function delegates
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void CharArrayCallback(IntPtr charArray);

#if ANDROID || IOS || VISIONOS
        // Static P/Invoke declarations for mobile platforms
        public LlamaLib() { }

        // Base LLM functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Tokenize")]
        public static extern IntPtr LLM_Tokenize(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Detokenize")]
        public static extern IntPtr LLM_Detokenize(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Embeddings")]
        public static extern IntPtr LLM_Embeddings(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Completion")]
        public static extern IntPtr LLM_Completion(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData, CharArrayCallback callback);

        // LLMLocal functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Slot")]
        public static extern IntPtr LLM_Slot(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Cancel")]
        public static extern void LLM_Cancel(IntPtr llm, int idSlot);

        // LLMProvider functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Lora_Weight")]
        public static extern IntPtr LLM_Lora_Weight(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Lora_List")]
        public static extern IntPtr LLM_Lora_List(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Delete")]
        public static extern void LLM_Delete(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Start")]
        public static extern void LLM_Start(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Started")]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool LLM_Started(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Stop")]
        public static extern void LLM_Stop(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Start_Server")]
        public static extern void LLM_Start_Server(IntPtr llm, 
            [MarshalAs(UnmanagedType.LPStr)] string host, 
            int port, 
            [MarshalAs(UnmanagedType.LPStr)] string apiKey);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Stop_Server")]
        public static extern void LLM_Stop_Server(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Join_Service")]
        public static extern void LLM_Join_Service(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Join_Server")]
        public static extern void LLM_Join_Server(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Set_SSL")]
        public static extern void LLM_Set_SSL(IntPtr llm, 
            [MarshalAs(UnmanagedType.LPStr)] string sslCert, 
            [MarshalAs(UnmanagedType.LPStr)] string sslKey);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Status_Code")]
        public static extern int LLM_Status_Code(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Status_Message")]
        public static extern IntPtr LLM_Status_Message(IntPtr llm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLM_Embedding_Size")]
        public static extern int LLM_Embedding_Size(IntPtr llm);

        // LLMRuntime functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMRuntime_Construct")]
        public static extern IntPtr LLMRuntime_Construct(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads = -1,
            int numGpuLayers = 0,
            int numParallel = 1,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention = false,
            int contextSize = 4096,
            int batchSize = 2048,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly = false,
            IntPtr loraPaths = default,
            int loraPathCount = 0,
            [MarshalAs(UnmanagedType.LPStr)] string path = "");

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMRuntime_From_Command")]
        public static extern IntPtr LLMRuntime_From_Command(
            [MarshalAs(UnmanagedType.LPStr)] string command,
            [MarshalAs(UnmanagedType.LPStr)] string path = "");

        // LLMService functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMService_Construct")]
        public static extern IntPtr LLMService_Construct(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads = -1,
            int numGpuLayers = 0,
            int numParallel = 1,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention = false,
            int contextSize = 4096,
            int batchSize = 2048,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly = false,
            int loraPathCount = 0,
            IntPtr loraPaths = default);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMService_From_Command")]
        public static extern IntPtr LLMService_From_Command([MarshalAs(UnmanagedType.LPStr)] string paramsString);

        // LLMClient functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMClient_Construct")]
        public static extern IntPtr LLMClient_Construct(IntPtr llm);

        // LLMRemoteClient functions
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "LLMRemoteClient_Construct")]
        public static extern IntPtr LLMRemoteClient_Construct(
            [MarshalAs(UnmanagedType.LPStr)] string url,
            int port);

        // Runtime functions for mobile platforms
        public static bool Has_GPU_Layers(string command)
        {
            return false;
        }

        public static string Available_Architectures(bool gpu)
        {
            return "";
        }

#else
        // Desktop platform implementation with dynamic loading

        static LlamaLib()
        {
            try
            {
                runtimeLibraryHandle = LibraryLoader.LoadLibrary(GetRuntimeLibraryPath());
                if (runtimeLibraryHandle != IntPtr.Zero)
                {
                    Has_GPU_Layers = LibraryLoader.GetSymbolDelegate<Has_GPU_Layers_Delegate>(runtimeLibraryHandle, "Has_GPU_Layers");
                    Available_Architectures = LibraryLoader.GetSymbolDelegate<Available_Architectures_Delegate>(runtimeLibraryHandle, "Available_Architectures");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load runtime library: {ex.Message}");
                throw;
            }
        }

        public LlamaLib(bool gpu = false)
        {
            LoadLibraries(gpu);
        }

        static string GetRuntimeLibraryPath()
        {
#if LINUX
            string libName = "libllamalib_linux_runtime.so";
#elif MACOS
            string libName = "libllamalib_macos_runtime.dylib";
#else
            string libName = "llamalib_windows_runtime.dll";
#endif
            return Path.Combine(libraryBasePath, libName);
        }

        private void LoadLibraries(bool gpu)
        {
            if (Available_Architectures == null)
            {
                throw new InvalidOperationException("Runtime library not loaded. Cannot determine available architectures.");
            }

            string architecturesString = Available_Architectures(gpu);
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
                    string libraryPath = Path.Combine(libraryBasePath, library.Trim());
                    libraryHandle = LibraryLoader.LoadLibrary(libraryPath);
                    if (libraryHandle == IntPtr.Zero)
                    {
                        Console.WriteLine($"Failed to load library {library}.");
                        continue;
                    }
                    LoadFunctionPointers();
                    architecture = library.Trim();
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load library {library}: {ex.Message}.");
                    lastException = ex;
                    continue;
                }
            }

            // If we get here, no library was successfully loaded
            throw new InvalidOperationException($"Failed to load any library. Available libraries: {string.Join(", ", libraries)}", lastException);
        }

        private void LoadFunctionPointers()
        {
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
            LLMRuntime_Construct = LibraryLoader.GetSymbolDelegate<LLMRuntime_Construct_Delegate>(libraryHandle, "LLMRuntime_Construct");
            LLMRuntime_From_Command = LibraryLoader.GetSymbolDelegate<LLMRuntime_From_Command_Delegate>(libraryHandle, "LLMRuntime_From_Command");
            LLMService_Construct = LibraryLoader.GetSymbolDelegate<LLMService_Construct_Delegate>(libraryHandle, "LLMService_Construct");
            LLMService_From_Command = LibraryLoader.GetSymbolDelegate<LLMService_From_Command_Delegate>(libraryHandle, "LLMService_From_Command");
            LLMClient_Construct = LibraryLoader.GetSymbolDelegate<LLMClient_Construct_Delegate>(libraryHandle, "LLMClient_Construct");
            LLMRemoteClient_Construct = LibraryLoader.GetSymbolDelegate<LLMRemoteClient_Construct_Delegate>(libraryHandle, "LLMRemoteClient_Construct");
        }

        // Delegate definitions for desktop platforms
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate string Available_Architectures_Delegate([MarshalAs(UnmanagedType.I1)] bool gpu);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate bool Has_GPU_Layers_Delegate([MarshalAs(UnmanagedType.LPStr)] string command);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Tokenize_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Detokenize_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Embeddings_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLM_Completion_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string jsonData, CharArrayCallback callback);
        
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
        public delegate void LLM_Start_Server_Delegate(IntPtr llm, [MarshalAs(UnmanagedType.LPStr)] string host, int port, [MarshalAs(UnmanagedType.LPStr)] string apiKey);
        
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
        public delegate IntPtr LLMRuntime_Construct_Delegate(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads,
            int numGpuLayers,
            int numParallel,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention,
            int contextSize,
            int batchSize,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly,
            IntPtr loraPaths,
            int loraPathCount,
            [MarshalAs(UnmanagedType.LPStr)] string path);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMRuntime_From_Command_Delegate([MarshalAs(UnmanagedType.LPStr)] string command, [MarshalAs(UnmanagedType.LPStr)] string path);
        
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LLMService_Construct_Delegate(
            [MarshalAs(UnmanagedType.LPStr)] string modelPath,
            int numThreads,
            int numGpuLayers,
            int numParallel,
            [MarshalAs(UnmanagedType.I1)] bool flashAttention,
            int contextSize,
            int batchSize,
            [MarshalAs(UnmanagedType.I1)] bool embeddingOnly,
            IntPtr loraPaths,
            int loraPathCount);
        
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
        public LLMRuntime_Construct_Delegate LLMRuntime_Construct;
        public LLMRuntime_From_Command_Delegate LLMRuntime_From_Command;
        public LLMService_Construct_Delegate LLMService_Construct;
        public LLMService_From_Command_Delegate LLMService_From_Command;
        public LLMClient_Construct_Delegate LLMClient_Construct;
        public LLMRemoteClient_Construct_Delegate LLMRemoteClient_Construct;
#endif

        public static string GetLibraryBasePath()
        {
            // Get the directory where the executable is located
            string baseDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            
#if LINUX
            string os = "linux";
#elif MACOS
            string os = "macos";
#else
            string os = "windows";
#endif

            string[] librariesDirNames = new string[] { "runtimes", "libraries", "libs" };
            foreach (string librariesDirName in librariesDirNames)
            {
                string librariesPath = Path.Combine(baseDir, librariesDirName, os);
                if (Directory.Exists(librariesPath)) return librariesPath;
            }

            return baseDir;
        }

        public void Dispose()
        {
#if !ANDROID && !IOS && !VISIONOS
            if (libraryHandle != IntPtr.Zero)
            {
                LibraryLoader.FreeLibrary(libraryHandle);
                libraryHandle = IntPtr.Zero;
            }
            if (runtimeLibraryHandle != IntPtr.Zero)
            {
                LibraryLoader.FreeLibrary(runtimeLibraryHandle);
            }
#endif
        }

        ~LlamaLib()
        {
            Dispose();
        }
    }
}