using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;

namespace LlamaLibExamples
{
    class Program
    {
        static string previousText = "";
        static void StreamingCallback(string text)
        {
            Console.Write(text.Substring(previousText.Length));
            previousText = text;
        }

        static void Main(string[] args)
        {
            string prompt = "The capital of";

            LLMService llmService = new LLMService("model.gguf", numGpuLayers:10);
            llmService.Start();

            // Optional: limit the amount of tokens that we can predict so that it doesn't produce text forever (some models do)
            llmService.SetCompletionParameters(new JObject { { "n_predict", 20 } });

            Console.WriteLine("----------------------- tokenize -----------------------");
            List<int> tokens = llmService.Tokenize(prompt);
            Console.Write("tokens: ");
            foreach (int token in tokens)
            {
                Console.Write($"{token} ");
            }
            Console.WriteLine();

            Console.WriteLine("\n----------------------- detokenize -----------------------");
            string detokenizeResponse = llmService.Detokenize(tokens);
            Console.WriteLine($"prompt: {detokenizeResponse}");

            Console.WriteLine("\n----------------------- completion (streaming) -----------------------");
            Console.Write("response: ");
            llmService.Completion(prompt, StreamingCallback);
            Console.WriteLine();

            Console.WriteLine("\n----------------------- completion (no streaming) -----------------------");
            string completionResponse = llmService.Completion(prompt);
            Console.WriteLine($"response: {completionResponse}");
        }
    }
}