import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import matter from 'gray-matter';
import type { MCPToolRegistry } from './mcp-tool-registry.js';

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
}

/**
 * Agent Metadata from frontmatter
 */
interface AgentMetadata {
  name: string;
  description: string;
  model: string;
  tools?: string[] | ['*'];
  handoffs?: Array<{
    agent: string;
    label: string;
    condition?: string;
  }>;
}

interface AgentDefinition {
  metadata: AgentMetadata;
  instructions: string;
  filePath: string;
}

export class AgentRegistry {
  private agents: Map<string, AgentDefinition> = new Map();

  constructor(
    private agentsDirectory: string,
    private toolRegistry: MCPToolRegistry
  ) {}

  async loadAll(): Promise<void> {
    console.debug(`[AgentRegistry] Loading agents from ${this.agentsDirectory}`);

    const files = readdirSync(this.agentsDirectory);
    const agentFiles = files.filter(f => f.endsWith('.agent.md'));

    console.debug(`[AgentRegistry] Found ${agentFiles.length} agent files`);

    for (const file of agentFiles) {
      const filePath = join(this.agentsDirectory, file);
      try {
        const definition = this.loadAgent(filePath);
        this.agents.set(definition.metadata.name, definition);
        console.debug(`[AgentRegistry] Loaded agent: ${definition.metadata.name}`);
      } catch (error) {
        console.error(`[AgentRegistry] Failed to load ${file}:`, error);
      }
    }

    console.debug(`[AgentRegistry] Successfully loaded ${this.agents.size} agents`);
  }

  private loadAgent(filePath: string): AgentDefinition {
    const content = readFileSync(filePath, 'utf-8');
    
    // Parse frontmatter
    const parsed = matter(content);
    const metadata = parsed.data as AgentMetadata;
    const instructions = parsed.content.trim();

    // Validate required fields
    if (!metadata.name) {
      throw new Error(`Agent file ${filePath} missing required field: name`);
    }
    if (!metadata.description) {
      throw new Error(`Agent file ${filePath} missing required field: description`);
    }
    // Note: model is optional - defaults to provider's default model if not specified

    return {
      metadata,
      instructions,
      filePath,
    };
  }

  getDefinition(agentName: string): AgentDefinition | undefined {
    return this.agents.get(agentName);
  }

  listAgents(): Array<{
    name: string;
    description: string;
    model: string;
    tools: string[] | ['*'];
    handoffs: string[];
  }> {
    return Array.from(this.agents.values()).map(def => ({
      name: def.metadata.name,
      description: def.metadata.description,
      model: def.metadata.model,
      tools: def.metadata.tools || ['*'],
      handoffs: (def.metadata.handoffs || []).map(h => h.agent),
    }));
  }
}
