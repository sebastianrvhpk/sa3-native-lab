import { Box, Braces, CircleDot, FileAudio } from "lucide-react";

import type { ArtifactRecord } from "./types";

export function ArtifactBadge({ artifact }: { artifact: ArtifactRecord }) {
  return (
    <span className={`artifact-badge ${artifact.kind}`}>
      <ArtifactIcon artifact={artifact} />
      {artifact.kind}
    </span>
  );
}

export function ArtifactIcon({ artifact }: { artifact: ArtifactRecord }) {
  if (artifact.kind === "audio") return <FileAudio size={18} />;
  if (artifact.kind === "latent") return <Braces size={18} />;
  if (artifact.kind === "bundle") return <Box size={18} />;
  return <CircleDot size={18} />;
}
