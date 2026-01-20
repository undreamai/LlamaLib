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
            // Create the underlying LLM service
            LLMService llmService = new LLMService("model.gguf", numGpuLayers:10);
            llmService.Start();

            // Create an agent with a system prompt
            string systemPrompt = "You are a helpful AI assistant. Be concise and friendly.";
            LLMAgent agent = new LLMAgent(llmService, systemPrompt);

            // First conversation turn
            Console.WriteLine("----------------------- First Turn -----------------------");
            string userMessage1 = "Hello! What's your name?";
            Console.WriteLine($"User: {userMessage1}");
            Console.Write("Assistant: ");
            string response1 = agent.Chat(userMessage1, true, StreamingCallback);
            Console.WriteLine();

            // Second conversation turn (maintains context)
            Console.WriteLine("\n----------------------- Second Turn -----------------------");
            string userMessage2 = "How are you today?";
            Console.WriteLine($"User: {userMessage2}");
            Console.Write("Assistant: ");
            previousText = "";
            string response2 = agent.Chat(userMessage2, true, StreamingCallback);
            Console.WriteLine();

            // Show conversation history
            Console.WriteLine("\n----------------------- Conversation History -----------------------");
            Console.WriteLine($"History size: {agent.GetHistorySize()} messages");
            List<ChatMessage> history = agent.GetHistory();
            foreach (ChatMessage msg in history)
            {
                Console.WriteLine($"{msg.role}: {msg.content}");
            }

            // Save conversation history
            Console.WriteLine("\n----------------------- Save/Load History -----------------------");
            string historyFile = "conversation_history.json";
            agent.SaveHistory(historyFile);
            Console.WriteLine($"History saved to: {historyFile}");

            // Clear history and reload
            agent.ClearHistory();
            Console.WriteLine($"History cleared. Size: {agent.GetHistorySize()}");

            agent.LoadHistory(historyFile);
            Console.WriteLine($"History loaded. Size: {agent.GetHistorySize()}");

            // Demonstrate manual message addition
            Console.WriteLine("\n----------------------- Manual Message Addition -----------------------");
            agent.AddUserMessage("This is a manually added user message");
            agent.AddAssistantMessage("This is a manually added assistant response");
            Console.WriteLine($"Added manual messages. New history size: {agent.GetHistorySize()}");

            // Cleanup
            llmService.Stop();
            agent.Dispose();
            llmService.Dispose();
        }
    }
}