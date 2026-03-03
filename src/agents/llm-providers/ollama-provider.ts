/**
 * Ollama LLM Provider
 * 
 * Local LLM via Ollama API (OpenAI-compatible).
 * Useful for development and privacy-first deployments.
 */

import OpenAI from 'openai';
import { LLMProvider, LLMConfig, LLMMessage, LLMResponse } from '../llm-provider.js';

export class OllamaProvider extends LLMProvider {
  private client: OpenAI;

  constructor(config: LLMConfig) {
    super(config);

    const baseUrl = config.baseUrl || 'http://localhost:11434/v1';

    this.client = new OpenAI({
      apiKey: 'ollama', // Required but ignored by Ollama
      baseURL: baseUrl,
    });
  }

  async complete(
    messages: LLMMessage[],
    tools?: Array<{
      name: string;
      description: string;
      inputSchema: Record<string, any>;
    }>
  ): Promise<LLMResponse> {
    const response = await this.client.chat.completions.create({
      model: this.config.model,
      messages: messages.map(msg => ({
        role: msg.role,
        content: msg.content,
      })) as any,
      // Note: Most Ollama models don't support function calling yet
      // Tools parameter might be ignored or cause errors
      tools: undefined,
      temperature: this.config.temperature ?? 0.7,
      max_tokens: this.config.maxTokens ?? 2048,
      top_p: this.config.topP ?? 1.0,
    });

    const firstChoice = response.choices[0];
    const content = firstChoice.message.content || '';

    // Ollama typically doesn't support tool calling yet
    // We could parse function calls from text content in the future
    return {
      content,
      toolCalls: [],
      stopReason: 'end_turn',
    };
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await this.client.chat.completions.create({
        model: this.config.model,
        messages: [
          {
            role: 'user',
            content: 'ping',
          },
        ],
        max_tokens: 5,
      });
      return !!response;
    } catch (error) {
      console.error('[OllamaProvider] Availability check failed:', error);
      return false;
    }
  }

  getProviderName(): string {
    return 'Ollama';
  }
}
