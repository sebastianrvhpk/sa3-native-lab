import { Box, Brain, GitFork, Heart, Route, Sparkles, Waves, Wand2 } from "lucide-react";

import type { GestureId, GestureOption } from "./gestureModel";

interface GestureStripProps {
  gestures: readonly GestureOption[];
  activeId: GestureId;
  onSelect: (id: GestureId) => void;
}

export function GestureStrip({ gestures, activeId, onSelect }: GestureStripProps) {
  return (
    <section className="gesture-strip" aria-label="Gestures">
      <div className="gesture-strip-head">
        <span className="eyebrow">Gestures</span>
        <strong>Choose the next move</strong>
      </div>
      <div className="gesture-buttons">
        {gestures.map((gesture) => (
          <button
            key={gesture.id}
            type="button"
            className={activeId === gesture.id ? "active" : ""}
            data-gesture={gesture.id}
            onClick={() => onSelect(gesture.id)}
            aria-pressed={activeId === gesture.id}
            title={gesture.disabledReason ?? gesture.shortIntent}
          >
            <GestureIcon id={gesture.id} />
            <span>{gesture.label}</span>
            <small>{gesture.available ? gesture.shortIntent : gesture.disabledReason}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function GestureIcon({ id }: { id: GestureId }) {
  if (id === "make") return <Wand2 size={18} />;
  if (id === "continue") return <Route size={18} />;
  if (id === "vary") return <Sparkles size={18} />;
  if (id === "steer") return <Brain size={18} />;
  if (id === "borrow_texture") return <GitFork size={18} />;
  if (id === "encode") return <Box size={18} />;
  if (id === "decode") return <Waves size={18} />;
  if (id === "morph") return <Sparkles size={18} />;
  return <Heart size={18} />;
}
