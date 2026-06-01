import { parseNumberList, stringValue, type RecipeValue } from "./recipeFormModel";
import type { ArtifactRecord } from "./types";
import type { LatentOperatorMode } from "./workbenchConfigs";

export interface LatentRegionDescriptor {
  title: string;
  detail: string;
  chips: string[];
  warning: string | null;
}

export function latentRegionDescriptor({
  operatorMode,
  form,
  selectedArtifact,
}: {
  operatorMode: LatentOperatorMode;
  form: Record<string, RecipeValue>;
  selectedArtifact: ArtifactRecord | null;
}): LatentRegionDescriptor | null {
  if (operatorMode === "latent.graft" || operatorMode === "latent.renoise") {
    return channelMaskDescriptor(operatorMode, form, selectedArtifact);
  }
  if (operatorMode === "latent.blur") {
    return blurDescriptor(form, selectedArtifact);
  }
  if (operatorMode === "latent.dsp") {
    return dspDescriptor(form);
  }
  return null;
}

function channelMaskDescriptor(
  operatorMode: "latent.graft" | "latent.renoise",
  form: Record<string, RecipeValue>,
  selectedArtifact: ArtifactRecord | null,
): LatentRegionDescriptor {
  const mode = stringValue(form.mode) || "random_channels";
  const explicitChannels = stringValue(form.channels) ? parseNumberList(form.channels ?? "").map((item) => Math.trunc(item)) : [];
  const start = stringValue(form.start_channel);
  const span = stringValue(form.block_size);
  const fraction = stringValue(form.fraction) || "0.25";
  const shape = latentShapeLabel(selectedArtifact);
  const mask = explicitChannels.length
    ? `${explicitChannels.length} exact channel${explicitChannels.length === 1 ? "" : "s"}`
    : start && span
    ? `block ${start} + ${span}`
    : `${fraction} channel fraction`;
  return {
    title: operatorMode === "latent.graft" ? "Channel mask for borrowed texture" : "Channel mask for renoise",
    detail: `${humanMode(mode)} over ${mask}${shape ? ` in ${shape}` : ""}.`,
    chips: [humanMode(mode), mask, operatorMode === "latent.graft" ? "donor blend" : "noise only"],
    warning: "This backend supports channel masks, not bounded time-region masks.",
  };
}

function blurDescriptor(
  form: Record<string, RecipeValue>,
  selectedArtifact: ArtifactRecord | null,
): LatentRegionDescriptor {
  const mode = stringValue(form.mode) || "temporal";
  const radius = Number(form.temporal_radius ?? 0);
  const seconds = latentSeconds(radius, selectedArtifact);
  return {
    title: "Temporal smear",
    detail: `${humanMode(mode)} uses a ${Number.isFinite(radius) ? radius : 0}-frame latent-time radius${seconds ? ` (~${seconds})` : ""}.`,
    chips: [humanMode(mode), `${Number.isFinite(radius) ? radius : 0} frames`, stringValue(form.temporal_direction) || "centered"],
    warning: "Blur is a global latent transform; it does not submit a start/end time mask.",
  };
}

function dspDescriptor(form: Record<string, RecipeValue>): LatentRegionDescriptor {
  const mode = stringValue(form.mode) || "gain";
  const needsDonor = mode === "fft_phase_blend" || mode === "fft_mag_phase_graft" || mode === "fft_phase_from_donor";
  return {
    title: "Latent-time DSP",
    detail: `${humanMode(mode)} changes the latent trajectory before decode or polish.`,
    chips: [humanMode(mode), `strength ${stringValue(form.strength) || "1"}`, needsDonor ? "needs donor latent" : "current latent"],
    warning: "FFT controls operate in latent time, not as waveform EQ.",
  };
}

function latentShapeLabel(artifact: ArtifactRecord | null) {
  const latent = artifact?.latent;
  if (!latent?.shape.length) return "";
  const channels = latent.channel_first ? latent.shape[0] : latent.shape[1];
  const frames = latent.channel_first ? latent.shape[1] : latent.shape[0];
  return `${channels}ch x ${frames} frames`;
}

function latentSeconds(frames: number, artifact: ArtifactRecord | null) {
  const rate = artifact?.latent?.latent_rate;
  if (!rate || !Number.isFinite(frames) || frames <= 0) return "";
  const seconds = frames / rate;
  return `${seconds.toFixed(seconds >= 1 ? 1 : 2)}s`;
}

function humanMode(value: string) {
  return value.replaceAll("_", " ");
}
