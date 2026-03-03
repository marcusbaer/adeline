/**
 * OpenAI LLM Provider
 * 
 * Supports both Azure OpenAI and OpenAI API endpoints.
 * Handles streaming, tool calls, and structured output.
 */

import OpenAI from 'openai';
import { LLMProvider, LLMConfig, LLMMessage, LLMResponse } from '../llm-provider.js';

export class OpenAIProvider extends LLMProvider {
  private client: OpenAI;

  constructor(config: LLMConfig) {
    super(config);

    // Support both OpenAI and Azure OpenAI
    const isAzure = config.baseUrl?.includes('openai.azure.com');

    if (isAzure) {
      this.client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseUrl,
        defaultHeaders: {
          'api-key': config.apiKey,
        },
      } as any);
    } else {
      this.client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseUrl,
      });
    }
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
      tools: tools
        ? tools.map(tool => ({
            type: 'function' as const,
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.inputSchema,
            },
          }))
        : undefined,
      temperature: this.config.temperature ?? 0.7,
      max_tokens: this.config.maxTokens ?? 4096,
      top_p: this.config.topP ?? 1.0,
    });

    const firstChoice = response.choices[0];
    const content = firstChoice.message.content || '';
    const toolCalls: LLMResponse['toolCalls'] = [];

    // Extract tool calls from response
    if (firstChoice.message.tool_calls) {
      for (const toolCall of firstChoice.message.tool_calls) {
        if (toolCall.type === 'function') {
          toolCalls.push({
            id: toolCall.id,
            name: toolCall.function.name,
            arguments: JSON.parse(toolCall.function.arguments),
          });
        }
      }
    }

    return {
      content,
      toolCalls,
      stopReason: firstChoice.finish_reason === 'tool_calls' ? 'tool_use' : 'end_turn',
    };
  }

  async isAvailable(): Promise<boolean> {
    try {
      if (!this.config.apiKey) {
        return false;
      }
      // Quick validation with a minimal request
      const response = await this.client.models.retrieve(this.config.model);
      return !!response;
    } catch (error) {
      console.error('[OpenAIProvider] Availability check failed:', error);
      return false;
    }
  }

  getProviderName(): string {
    return 'OpenAI';
  }
}
