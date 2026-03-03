import { spawn, ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { EventEmitter } from 'node:events';

if (process.env.DEBUG === 'false') {
  console.debug = () => {};
}

/**
 * MCP Server Configuration from mcp.json
 */
interface MCPServerConfig {
  command: string;
  args: string[];
  env?: Record<string, string>;
}

/**
 * MCP Tool Definition
 */
interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
}

/**
 * MCP Configuration File Structure (matches mcp.json)
 */
interface MCPConfig {
  $schema?: string;
  version?: string;
  description?: string;
  servers: Record<string, {
    type: 'stdio' | 'http';
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    url?: string;
    tools?: MCPTool[];
  }>;
}

/**
 * JSON-RPC 2.0 Request
 */
interface JSONRPCRequest {
  jsonrpc: '2.0';
  id: number | string;
  method: string;
  params?: any;
}

/**
 * JSON-RPC 2.0 Response
 */
interface JSONRPCResponse {
  jsonrpc: '2.0';
  id: number | string;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

/**
 * Running MCP Server Instance
 */
interface MCPServerInstance {
  name: string;
  process: ChildProcess;
  config: {
    type: 'stdio' | 'http';
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    url?: string;
    tools?: MCPTool[];
  };
  tools: Set<string>;
  emitter: EventEmitter;
  pendingRequests: Map<number | string, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>;
}

/**
 * MCP Tool Registry
 * 
 * Manages MCP server processes and routes tool calls.
 * Loads configuration from mcp.json and spawns stdio-based servers.
 */
export class MCPToolRegistry {
  private config!: MCPConfig;
  private servers: Map<string, MCPServerInstance> = new Map();
  private tools: Map<string, MCPTool & { server: string }> = new Map();
  private requestId = 0;

  constructor(private configPath: string = 'mcp.json') {}

  /**
   * Load MCP configuration and start all servers
   */
  async initialize(): Promise<void> {
    console.debug(`Loading configuration from ${this.configPath}`);
    
    // Load and parse mcp.json
    const configFile = readFileSync(resolve(this.configPath), 'utf-8');
    this.config = JSON.parse(configFile);

    // Register static tools from config (if present, for backward compatibility)
    for (const [serverName, serverConfig] of Object.entries(this.config.servers)) {
      if (serverConfig.tools && Array.isArray(serverConfig.tools)) {
        for (const toolDef of serverConfig.tools) {
          this.tools.set(toolDef.name, { ...toolDef, server: serverName });
        }
      }
    }

    if (this.tools.size > 0) {
      console.debug(`Registered ${this.tools.size} static tools from config`);
    }

    // Start all MCP servers
    const serverNames = Object.keys(this.config.servers);
    console.debug(`Starting ${serverNames.length} MCP servers...`);

    for (const serverName of serverNames) {
      await this.startMCPServer(serverName);
    }

    console.debug(`All MCP servers started successfully with ${this.tools.size} total tools`);
  }

  /**
   * Start a single MCP server process
   */
  private async startMCPServer(serverName: string): Promise<void> {
    const serverConfig = this.config.servers[serverName];
    if (!serverConfig) {
      throw new Error(`Server "${serverName}" not found in mcp.json`);
    }

    // Check if server type is supported
    if (serverConfig.type === 'http') {
      console.warn(`HTTP servers not yet supported, skipping: ${serverName}`);
      return;
    }

    if (serverConfig.type !== 'stdio') {
      console.warn(`Unknown server type "${serverConfig.type}", skipping: ${serverName}`);
      return;
    }

    // Validate stdio server has required fields
    if (!serverConfig.command || !serverConfig.args) {
      console.warn(`Stdio server "${serverName}" missing command or args, skipping`);
      return;
    }

    console.debug(`Starting server: ${serverName}`);

    // Spawn MCP server as child process
    const serverProcess = spawn(serverConfig.command, serverConfig.args, {
      env: {
        ...process.env,
        ...serverConfig.env,
      },
      stdio: ['pipe', 'pipe', 'pipe'], // stdin, stdout, stderr
    });

    // Create server instance
    const instance: MCPServerInstance = {
      name: serverName,
      process: serverProcess,
      config: serverConfig,
      tools: new Set(),
      emitter: new EventEmitter(),
      pendingRequests: new Map(),
    };

    // Track which tools this server provides
    for (const [toolName, toolDef] of this.tools.entries()) {
      if (toolDef.server === serverName) {
        instance.tools.add(toolName);
      }
    }

    // Handle server stdout (JSON-RPC responses)
    let buffer = '';
    serverProcess.stdout?.on('data', (chunk: Buffer) => {
      buffer += chunk.toString();
      
      // Process line-delimited JSON
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const response: JSONRPCResponse = JSON.parse(line);
            this.handleServerResponse(instance, response);
          } catch (error) {
            console.error(`Failed to parse JSON from ${serverName}:`, line);
          }
        }
      }
    });

    // Handle server stderr (logs)
    serverProcess.stderr?.on('data', (chunk: Buffer) => {
      console.debug(`[${serverName}] ${chunk.toString().trim()}`);
    });

    // Handle server exit
    serverProcess.on('exit', (code, signal) => {
      console.warn(`Server ${serverName} exited with code ${code}, signal ${signal}`);
      this.servers.delete(serverName);
    });

    // Handle server errors
    serverProcess.on('error', (error) => {
      console.error(`Server ${serverName} error:`, error);
    });

    // Store server instance
    this.servers.set(serverName, instance);

    // Wait for server to be ready and discover tools
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Discover tools from server via MCP protocol
    try {
      await this.discoverServerTools(instance);
    } catch (error) {
      console.warn(`Failed to discover tools from ${serverName}:`, error);
    }

    console.debug(`Server ${serverName} started with ${instance.tools.size} tools`);
  }

  /**
   * Discover tools from an MCP server via tools/list request
   */
  private async discoverServerTools(instance: MCPServerInstance): Promise<void> {
    const requestId = ++this.requestId;

    // Send tools/list request
    const request: JSONRPCRequest = {
      jsonrpc: '2.0',
      id: requestId,
      method: 'tools/list',
      params: {},
    };

    console.debug(`Discovering tools from server ${instance.name}`);

    return new Promise((resolve, reject) => {
      // Set timeout (10 seconds)
      const timeout = setTimeout(() => {
        instance.pendingRequests.delete(requestId);
        reject(new Error(`Tool discovery timeout: ${instance.name}`));
      }, 10000);

      // Store pending request
      instance.pendingRequests.set(requestId, {
        resolve: (result: any) => {
          // Register discovered tools
          if (result && result.tools && Array.isArray(result.tools)) {
            for (const toolDef of result.tools) {
              const tool = {
                name: toolDef.name,
                description: toolDef.description || '',
                inputSchema: toolDef.inputSchema || { type: 'object', properties: {} },
                server: instance.name,
              };
              this.tools.set(toolDef.name, tool);
              instance.tools.add(toolDef.name);
            }
            console.debug(`Discovered ${result.tools.length} tools from ${instance.name}`);
          }
          resolve();
        },
        reject,
        timeout,
      });

      // Write request to server stdin
      instance.process.stdin?.write(JSON.stringify(request) + '\n');
    });
  }

  /**
   * Handle JSON-RPC response from MCP server
   */
  private handleServerResponse(instance: MCPServerInstance, response: JSONRPCResponse): void {
    const pending = instance.pendingRequests.get(response.id);
    if (!pending) {
      console.warn(`Received response for unknown request ID: ${response.id}`);
      return;
    }

    // Clear timeout
    clearTimeout(pending.timeout);
    instance.pendingRequests.delete(response.id);

    // Resolve or reject based on response
    if (response.error) {
      pending.reject(new Error(`MCP Error: ${response.error.message}`));
    } else {
      pending.resolve(response.result);
    }
  }

  /**
   * Call an MCP tool
   */
  async callTool(toolName: string, input: any): Promise<any> {
    const toolDef = this.tools.get(toolName);
    if (!toolDef) {
      throw new Error(`Tool "${toolName}" not found in registry`);
    }

    const serverName = toolDef.server;
    const serverInstance = this.servers.get(serverName);
    if (!serverInstance) {
      throw new Error(`Server "${serverName}" not running`);
    }

    // Generate unique request ID
    const requestId = ++this.requestId;

    // Build JSON-RPC request
    const request: JSONRPCRequest = {
      jsonrpc: '2.0',
      id: requestId,
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: input,
      },
    };

    console.debug(`Calling tool ${toolName} on server ${serverName}`);

    // Send request to server
    return new Promise((resolve, reject) => {
      // Set timeout (30 seconds)
      const timeout = setTimeout(() => {
        serverInstance.pendingRequests.delete(requestId);
        reject(new Error(`Tool call timeout: ${toolName}`));
      }, 30000);

      // Store pending request
      serverInstance.pendingRequests.set(requestId, { resolve, reject, timeout });

      // Write request to server stdin
      serverInstance.process.stdin?.write(JSON.stringify(request) + '\n');
    });
  }

  /**
   * Get tool definition
   */
  getTool(toolName: string): (MCPTool & { server: string }) | undefined {
    return this.tools.get(toolName);
  }

  /**
   * Get all registered tools
   */
  getAllTools(): Array<MCPTool & { server: string }> {
    return Array.from(this.tools.values());
  }

  /**
   * Get tools for a specific server
   */
  getToolsByServer(serverName: string): Array<MCPTool & { server: string }> {
    return Array.from(this.tools.values()).filter(tool => tool.server === serverName);
  }

  /**
   * Get MCP server names (for agent configuration)
   */
  getServerNames(): string[] {
    return Array.from(this.servers.keys());
  }

  /**
   * Get tools for a specific server in OpenAI SDK format
   */
  getToolsForOpenAIByServer(serverName: string): Array<{
    type: 'function';
    function: {
      name: string;
      description: string;
      parameters: any;
    };
  }> {
    return this.getToolsByServer(serverName).map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      },
    }));
  }

  /**
   * Get tools for OpenAI SDK (simplified format)
   */
  getToolsForOpenAI(): Array<{
    type: 'function';
    function: {
      name: string;
      description: string;
      parameters: any;
    };
  }> {
    return Array.from(this.tools.values()).map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      },
    }));
  }

  hasTool(toolName: string): boolean {
    return this.tools.has(toolName);
  }

  async shutdown(): Promise<void> {
    console.debug('Shutting down all MCP servers...');

    for (const [serverName, instance] of this.servers.entries()) {
      console.debug(`Stopping server: ${serverName}`);
      
      // Cancel all pending requests
      for (const [requestId, pending] of instance.pendingRequests.entries()) {
        clearTimeout(pending.timeout);
        pending.reject(new Error('Server shutdown'));
      }
      instance.pendingRequests.clear();

      // Kill server process
      instance.process.kill('SIGTERM');
      
      // Wait for graceful shutdown (max 5 seconds)
      await new Promise((resolve) => {
        const killTimeout = setTimeout(() => {
          instance.process.kill('SIGKILL');
          resolve(undefined);
        }, 5000);

        instance.process.once('exit', () => {
          clearTimeout(killTimeout);
          resolve(undefined);
        });
      });
    }

    this.servers.clear();
    console.debug('All servers stopped');
  }
}
