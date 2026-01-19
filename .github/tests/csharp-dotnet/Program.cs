using System;
using UndreamAI.LlamaLib;
using Newtonsoft.Json.Linq;

class Program
{
    static string previousText = "";
    static void StreamingCallback(string text)
    {
        Console.Write(text.Substring(previousText.Length));
        previousText = text;
    }


    static async Task Main()
    {
        LLMService llm = new LLMService("model.gguf", 1, -1, 5);
        llm.Start();
        llm.StartServer("0.0.0.0", 13333);

        // Create agent with system prompt
        LLMAgent agent1 = new LLMAgent(llm, "You are a helpful AI assistant. Be concise and friendly.");

        // With local LLMClient
        LLMClient localClient = new LLMClient(llm);
        LLMAgent agent2 = new LLMAgent(localClient, "You are a helpful assistant.");

        // With remote LLMClient
        LLMClient remoteClient = new LLMClient("http://localhost", 13333);
        LLMAgent agent3 = new LLMAgent(remoteClient, "You are a helpful assistant.");

        // Async Interact with the agent (streaming)
        foreach (LLMAgent agent in new LLMAgent[] { agent1, agent2, agent3 })
        {
            agent.SetCompletionParameters(new JObject { { "n_predict", 20 } });
            string response = await agent.ChatAsync("What is AI?");
            Console.WriteLine(response);
        }

    }
}
