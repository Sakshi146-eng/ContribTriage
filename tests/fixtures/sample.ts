// Fixture: TypeScript module for lexical parser tests.
// Includes interfaces, classes, functions, imports, and TODO comments.

import { readFileSync } from "fs";
import axios from "axios";
import type { Config } from "./types";

// TODO: Add request retry logic
// FIXME: Error handling is incomplete

export interface DataPayload {
  id: string;
  records: Record<string, unknown>[];
}

export class ApiClient {
  private baseUrl: string;

  constructor(config: Config) {
    this.baseUrl = config.apiUrl;
  }

  async fetchData(endpoint: string): Promise<DataPayload> {
    // BUG: timeout is not configurable
    const response = await axios.get(`${this.baseUrl}/${endpoint}`);
    return response.data as DataPayload;
  }

  async postData(endpoint: string, payload: unknown): Promise<void> {
    await axios.post(`${this.baseUrl}/${endpoint}`, payload);
  }
}

export class DataTransformer {
  transform(payload: DataPayload): DataPayload {
    return {
      ...payload,
      records: payload.records.filter((r) => Object.keys(r).length > 0),
    };
  }
}

export function loadConfig(filePath: string): Config {
  const raw = readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as Config;
}

export async function runPipeline(configPath: string): Promise<void> {
  const config = loadConfig(configPath);
  const client = new ApiClient(config);
  const data = await client.fetchData("records");
  const transformer = new DataTransformer();
  transformer.transform(data);
}
