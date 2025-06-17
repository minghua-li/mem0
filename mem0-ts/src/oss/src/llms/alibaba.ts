import OpenAI from "openai";
import { LLM, LLMResponse } from "./base";
import { LLMConfig, Message } from "../types";

export class AlibabaLLM implements LLM {
  private openai: OpenAI; // Assuming Alibaba uses a compatible client or the OpenAI client directly
  private model: string;

  constructor(config: LLMConfig) {
    if (!config.apiKey) {
      throw new Error("Alibaba API key is required.");
    }
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://dashscope.aliyuncs.com/compatible-mode/v1", // Potentially different for Alibaba
    });
    this.model = config.model || "qwen-turbo"; // Default Alibaba model
  }

  async generateResponse(
    messages: Message[],
    responseFormat?: { type: string },
    tools?: any[],
  ): Promise<string | LLMResponse> {
    const completion = await this.openai.chat.completions.create({
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant" | "tool";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      model: this.model,
      response_format: responseFormat as { type: "text" | "json_object" },
      ...(tools && { tools, tool_choice: "auto" }),
    });

    const response = completion.choices[0].message;

    if (response.tool_calls) {
      return {
        content: response.content || "",
        role: response.role,
        toolCalls: response.tool_calls.map((call) => ({
          id: call.id, // Added tool call id
          name: call.function.name,
          arguments: call.function.arguments,
        })),
      };
    }

    return response.content || "";
  }

  async generateChat(messages: Message[]): Promise<LLMResponse> {
    const completion = await this.openai.chat.completions.create({
      messages: messages.map((msg) => {
        const role = msg.role as "system" | "user" | "assistant" | "tool";
        return {
          role,
          content:
            typeof msg.content === "string"
              ? msg.content
              : JSON.stringify(msg.content),
        };
      }),
      model: this.model,
    });
    const response = completion.choices[0].message;
    return {
      content: response.content || "",
      role: response.role,
    };
  }
}
