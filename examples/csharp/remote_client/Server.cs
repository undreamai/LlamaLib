using System;
using System.Collections.Generic;
using System.Threading;
using UndreamAI.LlamaLib;

namespace LlamaLibExamples
{
    class Server
    {
        static void Main(string[] args)
        {
            string model = "model.gguf";
            int serverPort = 13333;

            // show debug messages
            LLM.Debug(1);

            Console.WriteLine("Starting LLM server...\n");
            LLMService llmServer = new LLMService(model);

            // start service and server
            llmServer.Start();
            llmServer.StartServer("", serverPort);

            // wait until server exit
            llmServer.JoinServer();
        }
    }
}