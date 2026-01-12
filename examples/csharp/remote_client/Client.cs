using System;
using System.Collections.Generic;
using System.Threading;
using UndreamAI.LlamaLib;

namespace LlamaLibExamples
{
    class Client
    {
        static void StreamingCallback(string text)
        {
            Console.Write(text);
        }

        static void Main(string[] args)
        {
            string serverUrl = "http://localhost";
            int serverPort = 13333;

            // Create a remote client that connects to the server
            LLMClient llmClient = new LLMClient(serverUrl, serverPort);

            string prompt = "you are an artificial intelligence assistant\n\n--- user: Hello, how are you?\n--- assistant";

            Console.WriteLine("----------------------- tokenize -----------------------");
            List<int> tokens = llmClient.Tokenize(prompt);
            Console.Write("tokens: ");
            foreach (int token in tokens)
            {
                Console.Write($"{token} ");
            }
            Console.WriteLine();

            Console.WriteLine("\n----------------------- detokenize -----------------------");
            string detokenizeResponse = llmClient.Detokenize(tokens);
            Console.WriteLine($"prompt: {detokenizeResponse}");

            Console.WriteLine("\n----------------------- completion (streaming) -----------------------");
            Console.Write("response: ");
            llmClient.Completion(prompt, StreamingCallback);
            Console.WriteLine();

            Console.WriteLine("\n----------------------- completion (no streaming) -----------------------");
            string completionResponse = llmClient.Completion(prompt);
            Console.WriteLine($"response: {completionResponse}");

            Console.WriteLine("\n----------------------- embeddings -----------------------");
            List<float> embeddings = llmClient.Embeddings(prompt);
            Console.Write("embeddings: ");
            int maxCount = Math.Min(embeddings.Count, 10);
            for (int i = 0; i < maxCount; i++)
            {
                Console.Write($"{embeddings[i]} ");
            }
            Console.WriteLine("...");
        }
    }
}