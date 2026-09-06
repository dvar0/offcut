#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { describeModel } from "./model.js";

const input = JSON.parse(readFileSync(0, "utf8"));
if (!input || typeof input !== "object" || Array.isArray(input)) {
  throw new Error("Input must be a JSON object");
}

process.stdout.write(`${JSON.stringify(describeModel(input.connection, input.model))}\n`);
