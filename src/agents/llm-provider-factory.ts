/**
 * LLM Provider Factory
 * 
 * Selects and configures the appropriate LLM provider based on environment variables.
 * Supports: OpenAI (including Azure OpenAI), Ollama.
 * 
 * Note: Azure OpenAI is configured via OpenAI provider with OPENAI_API_BASE set to Azure endpoint.
 */

import { LLMProvider, LLMConfig } from './llm-provider.js';
import { OpenAIProvider } from './llm-providers/openai-provider.js';
import { OllamaProvider } from './llm-providers/ollama-provider.js';

export type ProviderType = 'openai' | 'ollama';

/**
 * Configuration from environment variables
 */
export interface LLMProviderConfig {
  provider: ProviderType;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

/**
 * Factory for creating LLM provider instances
 */
export class LLMProviderFactory {
  /**
   * Parse model string with optional provider prefix
   * 
   * Supports format: [provider/]model
   * Examples:
   *   - "gpt-4o" → { provider: "openai", model: "gpt-4o" }
   *   - "openai/gpt-4o" → { provider: "openai", model: "gpt-4o" }
   *   - "ollama/llama2" → { provider: "ollama", model: "llama2" }
   *   - "ollama/llama3.1" → { provider: "ollama", model: "llama3.1" }
   */
  static parseModelString(modelString: string): { provider: ProviderType; model: string } {
    // Check if provider prefix exists
    const slashIndex = modelString.indexOf('/');
    
    if (slashIndex === -1) {
      // No provider prefix, default to openai
      return {
        provider: 'openai',
        model: modelString,
      };
    }

    const potentialProvider = modelString.substring(0, slashIndex).toLowerCase();
    
    // Validate provider
    if (!['openai', 'ollama'].includes(potentialProvider)) {
      // Not a valid provider, treat entire string as model name (default to openai)
      return {
        provider: 'openai',
        model: modelString,
      };
    }

    return {
      provider: potentialProvider as ProviderType,
      model: modelString.substring(slashIndex + 1),
    };
  }

  /**
   * Create provider from environment variables and config
   */
  static create(config?: Partial<LLMProviderConfig>): LLMProvider {
    let providerType: ProviderType;
    let model: string;

    // If model string is provided, parse it for provider prefix
    if (config?.model) {
      const parsed = this.parseModelString(config.model);
      providerType = config?.provider || parsed.provider;
      model = parsed.model;
    } else {
      providerType = config?.provider || this.detectProvider();
      model = this.getModelForProvider(providerType);
    }

    console.log(`[LLMProviderFactory] Creating ${providerType} provider with model: ${model}`);

    // Get global LLM parameters from environment (fallback)
    const globalTemperature = process.env.LLM_TEMPERATURE ? parseFloat(process.env.LLM_TEMPERATURE) : undefined;
    const globalMaxTokens = process.env.LLM_MAX_TOKENS ? parseInt(process.env.LLM_MAX_TOKENS, 10) : undefined;
    const globalTopP = process.env.LLM_TOP_P ? parseFloat(process.env.LLM_TOP_P) : undefined;

    const llmConfig: LLMConfig = {
      model,
      apiKey: config?.apiKey || this.getApiKeyForProvider(providerType),
      baseUrl: config?.baseUrl || this.getBaseUrlForProvider(providerType),
      temperature: config?.temperature ?? globalTemperature,
      maxTokens: config?.maxTokens ?? globalMaxTokens,
      topP: config?.topP ?? globalTopP,
    };

    switch (providerType) {
      case 'openai':
        return new OpenAIProvider(llmConfig);

      case 'ollama':
        return new OllamaProvider(llmConfig);

      default:
        // Fallback to OpenAI for unknown providers
        console.warn(`[LLMProviderFactory] Unknown provider '${providerType}', defaulting to OpenAI`);
        return new OpenAIProvider(llmConfig);
    }
  }

  /**
   * Get the default model for the configured provider
   */
  static getDefaultModel(): string {
    const provider = this.detectProvider();
    return this.getModelForProvider(provider);
  }

  /**
   * Auto-detect provider from environment variables
   */
  private static detectProvider(): ProviderType {
    // Priority order
    if (process.env.OPENAI_API_KEY) {
      return 'openai';
    }
    if (process.env.OLLAMA_BASE_URL) {
      return 'ollama';
    }

    // Default to OpenAI if nothing is configured
    console.warn('[LLMProviderFactory] No LLM provider configured, defaulting to OpenAI');
    return 'openai';
  }

  /**
   * Get API key for provider
   */
  private static getApiKeyForProvider(provider: ProviderType): string | undefined {
    switch (provider) {
      case 'openai':
        return process.env.OPENAI_API_KEY;
      case 'ollama':
        return undefined; // Ollama doesn't require API key
      default:
        return undefined;
    }
  }

  /**
   * Get base URL for provider
   */
  private static getBaseUrlForProvider(provider: ProviderType): string | undefined {
    switch (provider) {
      case 'openai':
        return process.env.OPENAI_API_BASE;
      case 'ollama':
        return process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';
      default:
        return undefined;
    }
  }

  /**
   * Get default model for provider
   */
  private static getModelForProvider(provider: ProviderType): string {
    switch (provider) {
      case 'openai':
        return process.env.OPENAI_MODEL || '';
      case 'ollama':
        return process.env.OLLAMA_MODEL || 'qwen3:4b';
      default:
        return '';
    }
  }

  /**
   * Validate provider configuration and availability
   */
  static async validateProvider(provider?: ProviderType): Promise<boolean> {
    const targetProvider = provider || this.detectProvider();
    try {
      const llmProvider = this.create({ provider });
      const available = await llmProvider.isAvailable();
      if (available) {
        console.log(`[LLMProviderFactory] ✅ ${provider} provider is available`);
      } else {
        console.warn(`[LLMProviderFactory] ⚠️  ${provider} provider is not available`);
      }
      return available;
    } catch (error) {
      console.error(`[LLMProviderFactory] ❌ ${provider} provider validation failed:`, error);
      return false;
    }
  }
}
