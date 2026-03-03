import "dotenv/config";
import {
  Agent,
  Runner,
  tool,
  user,
  system,
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

import OpenAI from 'openai';
import { MCPToolRegistry } from './mcp-tool-registry.js';
import { AgentRegistry } from './agent-loader.js';
import { AgentBootstrap } from './agent-bootstrap.js';
import { resolve, dirname } from 'node:path';
import * as path from "node:path";
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { cwd } from "node:process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const OLLAMA_DEFAULT_MODEL = "qwen3:4b";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'ollama';
const OPENAI_API_BASE = process.env.OPENAI_API_BASE || '';
const DEFAULT_MODEL = (OPENAI_API_KEY === 'ollama') ? OLLAMA_DEFAULT_MODEL : undefined;

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
  console.dir = () => {};
}

/**
 * Job Context passed to orchestrator agent
 */
export interface AgentsRunnerContext {
  executionId: string;
  filePath?: string;
}

export interface ExecutionResult {
  executionId: string;
  status: 'completed' | 'failed';
  startedAt: string;
  completedAt: string;
  finalOutput: string | {},
  durationSeconds: number;
  error?: {
    message: string;
    phase: string;
    timestamp: string;
  };
}

export class AgentsRunner {
  private toolRegistry!: MCPToolRegistry;
  private agentRegistry!: AgentRegistry;
  private agentBootstrap!: AgentBootstrap;
  private client!: OpenAI;
  private initialized = false;
  private dataContext!: any;

  // Return absolute paths to ensure they work from any working directory
  private get mcpConfigPath(): string {
    return resolve(cwd(), 'mcp.json');
  }

  private get agentsDirectory(): string {
    return resolve(cwd(), '.agents');
  }

  constructor() {}

  /**
   * Initialize tool and agent registries
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    console.debug('🚀 Initializing...\n');

    try {
      this.toolRegistry = new MCPToolRegistry(this.mcpConfigPath);
      await this.toolRegistry.initialize();
      console.debug(`✅ MCP Tool Registry initialized with ${this.toolRegistry.getAllTools().length} tools`);

      // Initialize Agent Registry
      this.agentRegistry = new AgentRegistry(
        this.agentsDirectory,
        this.toolRegistry
      );
      await this.agentRegistry.loadAll();
      console.debug(`✅ Agent Registry loaded ${this.agentRegistry.listAgents().length} agents`);

      // Initialize Agent Bootstrap (creates @openai/agents Agent instances)
      this.agentBootstrap = new AgentBootstrap(this.agentRegistry, this.toolRegistry);
      await this.agentBootstrap.bootstrapAll();
      console.debug(`✅ Agent Bootstrap complete with multi-agent support`);
      // Create OpenAI client for the runner

      this.client = new OpenAI({
        apiKey: OPENAI_API_KEY,
        baseURL: (OPENAI_API_KEY === 'ollama') ? "http://localhost:11434/v1/" : OPENAI_API_BASE,
      });

      setDefaultOpenAIClient(this.client as any);
      setTracingDisabled(true);
      this.initialized = true;
    } catch (error) {
      console.error('Initialization failed:', error);
      throw error;
    }
  }

  async runJob(context: AgentsRunnerContext): Promise<ExecutionResult> {
    if (!this.initialized) {
      throw new Error('Not initialized. Call initialize() first.');
    }

    const startTime = Date.now();
    console.debug(`Starting job ${context.executionId}`);

    try {
      // Get orchestrator agent (responsible for triage)
      this.dataContext = this.getContent(context);
      const orchestrator = this.agentBootstrap.getAgent('orchestrator');
      if (!orchestrator) {
        throw new Error('Orchestrator agent not found in bootstrap');
      }
      let history: AgentInputItem[] = [
        // system context message
        system(`## Your Context Data
To be able to fulfil your tasks, you need to have context data, which you get here.

## Context Understanding Instructions

In every context section (each is a separate sub headline here) the context itself is wrapped by a special marker, consisting of several chars ">" (begin) or "<" (end).
The begin and the end markers contain a specific string, so it should be clear, what will be wrapped by the marker.

For example, in case the content is:

>>>>>>>>>>>> HELLO_WORLD >>>>>>>>>>>>
<p>What is<br>
  you</p>
<<<<<<<<<<<< HELLO_WORLD <<<<<<<<<<<<

Then, the content of the marker with name "HELLO_WORLD" is:

<p>What is<br>
  you</p>

### Contexts

#### HTML Page Source Code

>>>>>>>>>>>> HTML_SOURCE >>>>>>>>>>>>
${this.dataContext}
<<<<<<<<<<<< HTML_SOURCE <<<<<<<<<<<<
        `),
        // initial message
        // user('')
      ];

      const runner = new Runner();

      // Run the orchestrator (agent and message are passed to run())
      const result = await runner.run(orchestrator, history, {
        stream: false,
        context: this.dataContext, // an object or file content
      });

      const endTime = Date.now();
      const durationSeconds = Math.floor((endTime - startTime) / 1000);

      console.debug(`Job ${context.executionId} completed in ${durationSeconds}s`);
      
      // Debug: Show complete output structure first
      console.debug('\n=== COMPLETE OUTPUT DUMP ===');
      console.dir(result.output, { depth: 5, colors: true });
      console.debug('=== END COMPLETE DUMP ===\n');
      
      // Debug: Show agent execution flow
      console.debug('\n=== AGENT EXECUTION FLOW ===');
      result.output.forEach((item: any, index: number) => {
        if (item.type === 'message') {
          const role = item.role;
          const status = item.status || 'unknown';
          const agentName = item.agent?.name || item.name || 'unknown';
          const contentTypes = item.content?.map((c: any) => c.type).join(', ') || 'none';
          const preview = item.content?.[0]?.text?.substring(0, 100) || '';
          
          console.debug(`\n[${index}] Message:`);
          console.debug(`    Role: ${role}`);
          console.debug(`    Agent: ${agentName}`);
          console.debug(`    Status: ${status}`);
          console.debug(`    Content Types: ${contentTypes}`);
          if (preview) {
            console.debug(`    Preview: ${preview}${preview.length === 100 ? '...' : ''}`);
          }
        } else if (item.type === 'function_call') {
          const functionName = item.function?.name || item.name || 'unknown';
          const agentName = item.agent?.name || item.name || 'unknown';
          const args = item.function?.arguments || item.arguments || '{}';
          let argsPreview = '';
          try {
            const parsedArgs = typeof args === 'string' ? JSON.parse(args) : args;
            argsPreview = JSON.stringify(parsedArgs, null, 2);
          } catch {
            argsPreview = String(args);
          }
          
          console.debug(`\n[${index}] Function Call:`);
          console.debug(`    Agent: ${agentName}`);
          console.debug(`    Function: ${functionName}`);
          console.debug(`    Arguments: ${argsPreview}`);
        } else if (item.type === 'function_call_result') {
          const functionName = item.function?.name || item.name || 'unknown';
          const agentName = item.agent?.name || item.name || 'unknown';
          const result = item.result || item.content || item.output || '';
          let resultPreview = String(result).substring(0, 200);
          if (String(result).length > 200) {
            resultPreview += '...';
          }
          
          console.debug(`\n[${index}] Function Result:`);
          console.debug(`    Agent: ${agentName}`);
          console.debug(`    Function: ${functionName}`);
          console.debug(`    Result: ${resultPreview}`);
        } else {
          console.debug(`\n[${index}] ${item.type}`);
          // Show any additional properties
          const keys = Object.keys(item).filter(k => k !== 'type');
          if (keys.length > 0) {
            console.debug(`    Properties: ${keys.join(', ')}`);
          }
        }
      });
      console.debug('\n=== END AGENT FLOW ===\n');

      const outputCompleted = await this.filterResponseFromOutput(result.output);
      const finalOutput = result.finalOutput || outputCompleted || "";

      return {
        executionId: context.executionId,
        status: 'completed',
        startedAt: new Date(startTime).toISOString(),
        completedAt: new Date(endTime).toISOString(),
        finalOutput,
        durationSeconds,
      };
    } catch (error: any) {
      const endTime = Date.now();
      const durationSeconds = Math.floor((endTime - startTime) / 1000);

      console.error(`Job ${context.executionId} failed:`, error);

      return {
        executionId: context.executionId,
        status: 'failed',
        startedAt: new Date(startTime).toISOString(),
        completedAt: new Date(endTime).toISOString(),
        finalOutput: "",
        durationSeconds,
        error: {
          message: error.message,
          phase: 'orchestration',
          timestamp: new Date().toISOString(),
        },
      };
    }
  }

  listAgents(): Array<{
    name: string;
    description: string;
    model: string;
    tools: string[] | ['*'];
    handoffs: string[];
  }> {
    if (!this.initialized) {
      throw new Error('Not initialized. Call initialize() first.');
    }

    return this.agentRegistry.listAgents();
  }

  async shutdown(): Promise<void> {
    if (!this.initialized) {
      return;
    }
    console.debug('Shutting down...');

    // Shutdown MCP servers
    await this.toolRegistry.shutdown();

    this.initialized = false;
    console.debug('Shutdown complete');
  }

  filterResponseFromOutput(output: object[]) {
    // console.dir(result.output, { depth: null });
    const completedMessages = output.filter(
      (item: any) => item.type === 'message' && item.status === 'completed'
    );
    const lastCompletedMessage = completedMessages[completedMessages.length - 1];

    let outputCompleted = '';
    // @ts-ignore
    if (lastCompletedMessage?.content) {
      // @ts-ignore
      const outputTextEntries = lastCompletedMessage.content.filter(
        (item: any) => item.type === 'output_text'
      );
      const lastOutputText = outputTextEntries[outputTextEntries.length - 1];
      outputCompleted = lastOutputText?.text || '';
    }
    return outputCompleted;
  }

  getContent(context: AgentsRunnerContext): string | object {
    // return { name: "John Doe", uid: 123 };
    if (!context.filePath) {
      return '';
    }
    const dir = context.filePath ? path.join(cwd(), context.filePath) : '';
    const fileContent = readFileSync(dir, { encoding: 'utf8', flag: 'r' });
    return fileContent;
  }
}
