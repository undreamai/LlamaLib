using System;
using System.Collections.Generic;
using System.Threading;
using UndreamAI.LlamaLib;

namespace LlamaLibExamples
{
    class Client
    {
        static string previousText = "";
        static void StreamingCallback(string text)
        {
            Console.Write(text.Substring(previousText.Length));
            previousText = text;
        }

        static void Main(string[] args)
        {
            string serverUrl = "http://localhost";
            string prompt = "Hello, how are you?";
            int serverPort = 13333;

            Console.WriteLine("*** Using client ***");
            // Create a remote client that connects to the server
            LLMClient llmClient = new LLMClient(serverUrl, serverPort);
            Console.WriteLine("----------------------- tokenize -----------------------");
            List<int> tokens = llmClient.Tokenize(prompt);
            Console.Write("tokens: ");
            foreach (int token in tokens)
            {
                Console.Write($"{token} ");
            }
            Console.WriteLine();

            // Create an agent that uses the remote client
            Console.WriteLine("*** Using agent ***");
            string systemPrompt = "You are a helpful AI assistant. Be concise and friendly.";
            LLMAgent agent = new LLMAgent(llmClient, systemPrompt);
            Console.WriteLine("\n----------------------- completion (streaming) using agent -----------------------");
            Console.WriteLine("User: " + prompt);
            Console.Write("Assistant: ");
            llmClient.Completion(prompt, StreamingCallback);
            Console.WriteLine();
        }
    }
}