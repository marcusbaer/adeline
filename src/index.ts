import { OpenAI } from "openai";
import {
  Agent,
  Runner,
  tool,
  user,
  setDefaultOpenAIClient,
  setTracingDisabled,
  ModelProvider,
  OpenAIChatCompletionsModel,
  RunContext,
  MCPServerStreamableHttp,
  MCPServerStdio,
  createMCPToolStaticFilter,
  run,
} from "@openai/agents";
import type { AgentInputItem } from "@openai/agents";
import { z } from "zod";
import * as path from "node:path";
import readline from "node:readline/promises";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const DEFAULT_MODEL = "qwen3:4b";

const client = new OpenAI({
  baseURL: "http://localhost:11434/v1/",
  apiKey: "ollama",
});

// conversationId requires OpenAI API endpoint
// const { id: conversationId } = await client.conversations.create({});

setDefaultOpenAIClient(client);
setTracingDisabled(true);

interface UserContext {
  name: string;
}

function buildInstructions(runContext: RunContext<UserContext>) {
  return runContext.context.name
    ? `The user's name is ${runContext.context.name}. Be extra friendly!`
    : "You are a helpful assistant";
}

const assistantAgent = new Agent({
  name: "Assistant",
  model: DEFAULT_MODEL,
  instructions: buildInstructions,
});

// const result = await run(
//   assistantAgent,
//   'Greet the user!',
// );

const mcpHttpServer = new MCPServerStreamableHttp({
  url: "https://gitmcp.io/openai/codex",
  name: "GitMCP Documentation Server",
});

// import.meta.filename, import.meta.dirname
const samplesDir = path.join(import.meta.dirname, "..", "sample_files");
const mcpFilesystemServer = new MCPServerStdio({
  name: "Filesystem MCP Server, via npx",
  fullCommand: `npx -y @modelcontextprotocol/server-filesystem ${samplesDir}`,
  timeout: 20000,
  // toolFilter: createMCPToolStaticFilter({
  //   allowed: ['safe_tool'],
  //   blocked: ['danger_tool'],
  // }),
});

interface UserInfo {
  name: string;
  uid: number;
}

const fetchUserAge = tool({
  name: "fetch_user_age",
  description: "Return the age of the current user",
  parameters: z.object({}),
  execute: async (
    _args,
    runContext?: RunContext<UserInfo>,
  ): Promise<string> => {
    return `User ${runContext?.context.name} is 47 years old`;
  },
});

// const sendEmail = tool({
//   name: 'sendEmail',
//   description: 'Send an email',
//   parameters: z.object({
//     to: z.string(),
//     subject: z.string(),
//     body: z.string(),
//   }),
//   needsApproval: async (_context, { subject }) => {
//     // check if the email is spam
//     return subject.includes('spam');
//   },
//   execute: async ({ to, subject, body }, args) => {
//     // send email
//   },
// });

const historyFunFact = tool({
  // The name of the tool will be used by the agent to tell what tool to use.
  name: "history_fun_fact",
  // The description is used to describe **when** to use the tool by telling it **what** it does.
  description: "Give a fun fact about a historical event",
  // This tool takes no parameters, so we provide an empty Zod Object.
  parameters: z.object({}),
  execute: async () => {
    // The output will be returned back to the Agent to use
    return "Sharks are older than trees.";
  },
});

const getWeather = tool({
  name: "get_weather",
  description: "Return the weather for a given city.",
  parameters: z.object({ city: z.string() }),
  needsApproval: async (_context, { city }) => {
    // forces approval to look up the weather in San Francisco
    return city === "San Francisco";
  },
  async execute({ city }) {
    return `The weather in ${city} is sunny.`;
  },
});

const userInfo: UserInfo = { name: "John", uid: 123 };

const userAgeAgent = new Agent<UserInfo>({
  name: "User Age Assistant",
  instructions: "You provide assistance with questions to the user's age.",
  model: DEFAULT_MODEL,
  tools: [fetchUserAge],
});

const weatherAgent = new Agent({
  name: "Weather bot",
  instructions: "You are a helpful weather bot.",
  model: DEFAULT_MODEL,
  modelSettings: {
    temperature: 0,
  },
  mcpServers: [mcpHttpServer],
  tools: [getWeather],
});

const historyTutorAgent = new Agent({
  name: "History Tutor",
  model: DEFAULT_MODEL,
  instructions:
    "You provide assistance with historical queries. Explain important events and context clearly.",
  // Adding the tool to the agent
  tools: [historyFunFact],
});

const mathTutorAgent = new Agent({
  name: "Math Tutor",
  model: DEFAULT_MODEL,
  instructions:
    "You provide help with math problems. Explain your reasoning at each step and include examples",
});

const fileSystemAgent = new Agent({
  name: "FS MCP Assistant",
  model: DEFAULT_MODEL,
  instructions:
    "Use the tools to read the filesystem and answer questions based on those files. If you are unable to find any files, you can say so instead of assuming they exist.",
  mcpServers: [mcpFilesystemServer],
});

// Using the Agent.create method to ensures type safety for the final output
const triageAgent = Agent.create({
  name: "Triage Agent",
  model: DEFAULT_MODEL,
  instructions:
    "You determine which agent to use based on the user's homework question",
  handoffs: [
    fileSystemAgent,
    historyTutorAgent,
    mathTutorAgent,
    userAgeAgent,
    weatherAgent,
  ],
});

async function runChat(actionPrompt = '') {
    let history: AgentInputItem[] = [
      // initial message
      user(actionPrompt),
    ];

    let userPrompt = "";
    while (userPrompt !== "/bye") {
      
      const result = await run(triageAgent, history, {
        // conversationId,
        stream: false,
        context: userInfo,
      });

      // update the history to the new output
      // history.push(user('How about now?'));

      // const usage = result.state.usage;
      // console.debug({
      //   requests: usage.requests,
      //   inputTokens: usage.inputTokens,
      //   outputTokens: usage.outputTokens,
      //   totalTokens: usage.totalTokens,
      // });
      // if (usage.requestUsageEntries) {
      //   for (const entry of usage.requestUsageEntries) {
      //     console.debug("request", {
      //       endpoint: entry.endpoint,
      //       inputTokens: entry.inputTokens,
      //       outputTokens: entry.outputTokens,
      //       totalTokens: entry.totalTokens,
      //     });
      //   }
      // }

      history = result.history;

      if (result.finalOutput) {
        userPrompt = await rl.question(`${result.finalOutput}\n\n----------\n\n`);
        history.push(user(userPrompt));
        // console.log(result.finalOutput);
        // console.log(result);
      }
    }

    // console.debug(result.history);
    process.stdout.write(JSON.stringify(history));
    rl.close();
    process.exit(0);
}

async function main() {
  try {
    await mcpHttpServer.connect();
    await mcpFilesystemServer.connect();
    // const actionPrompt = 'What is the capital of France and how it the current weather in that city?';
    // const actionPrompt = 'What is the current weather in San Francisco?';
    // const actionPrompt = "Read the files and list them.";
    // const actionPrompt = "What is the age of the user?";

    let stdinInput = "";
    process.stdin.setEncoding("utf8");

    process.stdin.on("data", (chunk) => {
      stdinInput += chunk;
    });

    process.stdin.on("end", async () => {
      await runChat(stdinInput);
    });

    if (process.stdin.isTTY) {
      const actionPrompt = await rl.question("What can I do for you?\n\n");
      if (!actionPrompt) {
        throw new Error("No input prompt provided");
      }
      await runChat(actionPrompt);
    }
  } finally {
    await mcpHttpServer.close();
    await mcpFilesystemServer.close();
  }
}

main().catch((err) => console.error(err));

// Checkout these helpful links:
// https://openai.github.io/openai-agents-js/guides/agents/
// https://openai.github.io/openai-agents-js/guides/running-agents/#the-agent-loop
// https://openai.github.io/openai-agents-js/guides/config/
// https://openai.github.io/openai-agents-js/guides/mcp/
// https://github.com/openai/openai-agents-js/blob/main/examples/model-providers/custom-example-provider.ts
