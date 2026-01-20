using System;
using System.Collections.Generic;
using UndreamAI.LlamaLib;

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
            LLMService llmService = new LLMService("model.gguf", embeddingOnly:true);
            llmService.Start();

            Console.WriteLine("\n----------------------- embeddings -----------------------");
            List<float> embeddings = llmService.Embeddings("this is the text I want to embed");
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