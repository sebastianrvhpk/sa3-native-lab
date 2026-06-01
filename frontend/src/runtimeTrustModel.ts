const secretPatterns: readonly [RegExp, string][] = [
  [/\b((?:HF_TOKEN|HUGGINGFACE_TOKEN|TOKEN|API_KEY|SECRET|PASSWORD)=)([^\s"'`]+)/gi, "$1[redacted]"],
  [/(--(?:hf-)?token(?:=|\s+))([^\s"'`]+)/gi, "$1[redacted]"],
  [/(--(?:api-key|secret|password)(?:=|\s+))([^\s"'`]+)/gi, "$1[redacted]"],
  [/((?:token|api[_-]?key|secret|password)["']?\s*[:=]\s*["']?)([^"',\s}]+)/gi, "$1[redacted]"],
  [/\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}/gi, "$1[redacted]"],
  [/\b(hf_[A-Za-z0-9_-]{8,})\b/g, "[redacted-token]"],
];

export function sanitizeRuntimeText(value: unknown, maxLength = 500): string {
  if (typeof value !== "string") return "";
  const trimmed = value.trim();
  if (!trimmed) return "";
  return redactSecrets(trimmed).slice(0, maxLength);
}

export function safeRuntimeLogLines(lines: readonly string[], limit = 12): string[] {
  return lines
    .slice(-limit)
    .map((line) => sanitizeRuntimeText(line, 1200))
    .filter(Boolean);
}

export function latestSafeRuntimeLine(lines: readonly string[]): string {
  const latest = [...lines].reverse().find((line) => line.trim() && !line.includes("[heartbeat]"));
  return sanitizeRuntimeText(latest, 180);
}

function redactSecrets(value: string): string {
  return secretPatterns.reduce((text, [pattern, replacement]) => text.replace(pattern, replacement), value);
}
