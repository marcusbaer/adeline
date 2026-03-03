/**
 * Agent Bootstrap
 * 
 * Creates @openai/agents Agent instances with proper tool integration and handoffs.
 * 
 * Tool Integration:
 * - Tools are managed by MCPToolRegistry
 * - This class converts MCP tools to @openai/agents format
 * - Tools are passed to agents for orchestration
 */

import { Agent, tool } from '@openai/agents';
import OpenAI from 'openai';
import { z } from 'zod';
import type { MCPToolRegistry } from './mcp-tool-registry.js';
import type { AgentRegistry } from './agent-loader.js';
import { LLMProviderFactory } from './llm-provider-factory.js';

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
}

/**
 * Bootstrap agents with tools and handoffs
 * 
 * Process:
 * 1. Load agent definitions
 * 2. Create OpenAI SDK clients with correct model/baseURL
 * 3. Get MCP tools from MCPToolRegistry
 * 4. Build agent instances with handoffs pointing to other agents
 * 5. Cache instances for reuse
 */
export class AgentBootstrap {
  private agents: Map<string, Agent> = new Map();
  private clients: Map<string, OpenAI> = new Map(); // One client per model/provider

  constructor(
    private agentRegistry: AgentRegistry,
    private toolRegistry: MCPToolRegistry
  ) {}

  /**
   * Bootstrap all agents from definitions
   */
  async bootstrapAll(): Promise<Map<string, Agent>> {
    console.debug('[AgentBootstrap] Bootstrapping all agents...');

    const definitions = this.agentRegistry.listAgents();

    // First pass: Create agents without handoffs (avoid circular dependencies)
    for (const def of definitions) {
      try {
        const agent = await this.bootstrapAgent(def.name, false);
        this.agents.set(def.name, agent);
        console.debug(`[AgentBootstrap] Bootstrapped agent: ${def.name}`);
      } catch (error) {
        console.error(`[AgentBootstrap] Failed to bootstrap ${def.name}:`, error);
      }
    }

    // Second pass: Add handoffs (now all agents exist)
    for (const def of definitions) {
      const agentDef = this.agentRegistry.getDefinition(def.name);
      if (agentDef?.metadata.handoffs && agentDef.metadata.handoffs.length > 0) {
        // Rebuild agent with handoffs
        try {
          const agent = await this.bootstrapAgent(def.name, true);
          this.agents.set(def.name, agent);
          console.debug(`[AgentBootstrap] Added handoffs to: ${def.name}`);
        } catch (error) {
          console.error(`[AgentBootstrap] Failed to add handoffs to ${def.name}:`, error);
        }
      }
    }

    console.debug(`[AgentBootstrap] Successfully bootstrapped ${this.agents.size} agents`);
    return this.agents;
  }

  /**
   * Bootstrap a single agent
   */
  private async bootstrapAgent(agentName: string, includeHandoffs: boolean): Promise<Agent> {
    const agentDef = this.agentRegistry.getDefinition(agentName);
    if (!agentDef) {
      throw new Error(`Agent definition not found: ${agentName}`);
    }

    // Use agent's model if specified, otherwise use provider's default
    const modelString = agentDef.metadata.model || LLMProviderFactory.getDefaultModel();

    // Get or create OpenAI client for this agent's model
    const client = await this.getClientForModel(modelString);

    // Get handoff agents (if including handoffs)
    const handoffs: Agent[] = [];
    if (includeHandoffs && agentDef.metadata.handoffs) {
      for (const handoff of agentDef.metadata.handoffs) {
        const handoffAgent = this.agents.get(handoff.agent);
        if (handoffAgent) {
          handoffs.push(handoffAgent);
        } else {
          console.warn(`[AgentBootstrap] Handoff agent not found: ${handoff.agent}`);
        }
      }
    }

    // Get tools for this agent
    const agentTools = this.buildToolsForAgent(agentDef.metadata.tools);

    // Create Agent instance with tools
    const agent = new Agent({
      name: agentDef.metadata.name,
      model: modelString,
      instructions: agentDef.instructions,
      handoffs: handoffs.length > 0 ? handoffs : undefined,
      tools: agentTools.length > 0 ? agentTools : undefined,
    });

    return agent;
  }

  /**
   * Get or create OpenAI client for model
   * 
   * Models may have provider prefix (e.g., "ollama/llama3.1")
   * This creates the appropriate client with correct baseURL
   */
  private async getClientForModel(modelString: string): Promise<OpenAI> {
    // Check cache first
    if (this.clients.has(modelString)) {
      return this.clients.get(modelString)!;
    }

    console.debug(`[AgentBootstrap] Creating client for model: ${modelString}`);

    // Parse model string for provider
    const modelSpec = LLMProviderFactory.parseModelString(modelString);

    // Create appropriate OpenAI client
    let client: OpenAI;

    if (modelSpec.provider === 'openai') {
      // Standard OpenAI or Azure
      const apiKey = process.env.OPENAI_API_KEY;
      const baseURL = process.env.OPENAI_API_BASE;

      if (!apiKey) {
        throw new Error('OPENAI_API_KEY environment variable is required for OpenAI models');
      }

      client = new OpenAI({
        apiKey,
        baseURL, // For Azure: https://{resource}.openai.azure.com/
      });
    } else if (modelSpec.provider === 'ollama') {
      // Ollama (OpenAI-compatible API)
      const baseURL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';

      client = new OpenAI({
        apiKey: 'ollama', // Dummy key for Ollama
        baseURL,
      });
    } else {
      throw new Error(`Unknown provider: ${modelSpec.provider}`);
    }

    // Cache client
    this.clients.set(modelString, client);
    return client;
  }

  /**
   * Get MCP tools from registry and format for @openai/agents
   * 
   * Retrieves tools from MCPToolRegistry and converts them to the format
   * expected by @openai/agents Agent constructor.
   */
  private getFormattedTools(
    toolNames?: string[] | ['*']
  ): Array<{ type: 'function'; function: any }> {
    // Get MCP tools from registry
    let mpcTools = [];

    if (!toolNames || toolNames.includes('*')) {
      // All tools
      mpcTools = this.toolRegistry.getAllTools();
    } else {
      // Specific tools
      mpcTools = (toolNames as string[])
        .map(name => this.toolRegistry.getTool(name))
        .filter((tool): tool is any => !!tool);
    }

    // Format tools for @openai/agents
    // The tools need the 'name' field to be present
    return mpcTools.map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      },
    }));
  }

  /**
   * Build tool objects for an agent
   * 
   * Converts MCP tools to @openai/agents tool() objects that can be
   * registered with the agent. If toolSpec contains server names
   * (e.g., ['poc-mcp-server-sak']), all tools from those servers will
   * be included.
   */
  private buildToolsForAgent(toolSpec?: string[] | ['*']): any[] {
    if (!toolSpec || toolSpec.includes('*')) {
      // Return all tools
      return this.toolRegistry.getAllTools().map(mcpTool => this.createToolObject(mcpTool));
    }

    const tools: any[] = [];

    // Check if toolSpec contains server names or tool names
    for (const item of toolSpec) {
      // Try it as a tool name first
      const toolDef = this.toolRegistry.getTool(item);
      if (toolDef) {
        // It's a tool name
        tools.push(this.createToolObject(toolDef));
      } else {
        // Try it as a server name - get all tools from this server
        const serverTools = this.toolRegistry.getToolsByServer(item);
        for (const mcpTool of serverTools) {
          tools.push(this.createToolObject(mcpTool));
        }
      }
    }

    return tools;
  }

  /**
   * Create a tool() object from an MCP tool definition
   */
  private createToolObject(mcpTool: any) {
    console.debug(`[AgentBootstrap] Creating tool object for: ${mcpTool.name}`, {
      description: mcpTool.description,
      inputSchema: mcpTool.inputSchema,
    });

    // For MCP tools, build a Zod schema from the inputSchema
    let schema = z.object({});

    if (mcpTool.inputSchema?.properties && typeof mcpTool.inputSchema.properties === 'object') {
      const props = mcpTool.inputSchema.properties;
      const required = mcpTool.inputSchema.required || [];
      const shapeObj: Record<string, any> = {};

      for (const [key] of Object.entries(props)) {
        // For all properties, accept string type (OpenAI will coerce the values)
        if (required.includes(key)) {
          shapeObj[key] = z.string();
        } else {
          shapeObj[key] = z.string().optional();
        }
      }

      schema = z.object(shapeObj);
    }

    return tool({
      name: mcpTool.name,
      description: mcpTool.description || 'MCP Tool',
      parameters: schema,
      execute: async (input: any) => {
        try {
          console.debug(`[AgentBootstrap] Executing tool: ${mcpTool.name}`, { input });
          const result = await this.toolRegistry.callTool(mcpTool.name, input || {});
          return result;
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          console.error(`[AgentBootstrap] Tool execution failed: ${mcpTool.name} - ${errorMsg}`);
          throw error;
        }
      },
    });
  }

  /**
   * Get an agent by name
   */
  getAgent(agentName: string): Agent | undefined {
    return this.agents.get(agentName);
  }

  /**
   * Get all bootstrapped agents
   */
  getAllAgents(): Map<string, Agent> {
    return this.agents;
  }
}
