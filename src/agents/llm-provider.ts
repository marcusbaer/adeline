/**
 * Abstract LLM Provider Interface
 * 
 * Defines the contract for LLM providers (OpenAI, Ollama, Replicate).
 * Handles streaming responses, tool calls, and error handling.
 */

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
}

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  toolCalls?: ToolCall[];
}

export interface LLMResponse {
  content: string;
  toolCalls: ToolCall[];
  stopReason: 'end_turn' | 'tool_use' | 'error';
}

export interface LLMConfig {
  apiKey?: string;
  baseUrl?: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  [key: string]: any;
}

/**
 * Abstract base class for LLM providers
 */
export abstract class LLMProvider {
  protected config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
  }

  /**
   * Get current configuration (for accessing model, etc.)
   */
  getConfig(): LLMConfig {
    return this.config;
  }

  /**
   * Complete a message and return response with potential tool calls
   */
  abstract complete(
    messages: LLMMessage[],
    tools?: Array<{
      name: string;
      description: string;
      inputSchema: Record<string, any>;
    }>
  ): Promise<LLMResponse>;

  /**
   * Check if provider is configured and available
   */
  abstract isAvailable(): Promise<boolean>;

  /**
   * Get provider name for logging
   */
  abstract getProviderName(): string;
}
