using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

namespace LlamaLibExamples
{
    class Program
    {
        static void StreamingCallback(string text)
        {
            Console.Write(text);
        }

        static void Main(string[] args)
        {
            string model = "model.gguf";

            // Create the underlying LLM service
            LLMService llmService = new LLMService(model);

            // Create a local client that wraps the service
            LLMClient llmClient = new LLMClient(llmService);

            llmService.Start();
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