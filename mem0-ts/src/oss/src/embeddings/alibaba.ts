import OpenAI from "openai";
import { Embedder } from "./base";
import { EmbeddingConfig } from "../types";

export class AlibabaEmbedder implements Embedder {
  private openai: OpenAI; // Assuming Alibaba uses a compatible client or the OpenAI client directly
  private model: string;

  constructor(config: EmbeddingConfig) {
    if (!config.apiKey) {
      throw new Error("Alibaba API key is required for embeddings.");
    }
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL || "https://dashscope.aliyuncs.com/compatible-mode/v1", // Potentially different for Alibaba
    });
    this.model = config.model || "text-embedding-v1"; // Default Alibaba embedding model, replace if necessary
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.openai.embeddings.create({
      model: this.model,
      input: texts,
    });
    return response.data.map((item) => item.embedding);
  }
}
