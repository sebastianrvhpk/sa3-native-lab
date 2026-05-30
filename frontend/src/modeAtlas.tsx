import { Database } from "lucide-react";

import type { NotebookMode, OperatorName } from "./types";
import { statusClass } from "./workbenchModel";

export function ModeAtlas({ modes, activeOperator }: { modes: NotebookMode[]; activeOperator: OperatorName }) {
  if (!modes.length) {
    return null;
  }
  return (
    <div className="mode-atlas">
      <div className="mode-atlas-head">
        <span>
          <Database size={16} />
          Colab Mode Atlas
        </span>
        <strong>{modes.length}</strong>
      </div>
      <div className="mode-atlas-list">
        {modes.map((mode) => {
          const active = mode.operators.includes(activeOperator);
          return (
            <article key={mode.mode_id} className={`mode-card ${active ? "active" : ""}`}>
              <div>
                <strong>
                  {mode.mode_id}. {mode.title}
                </strong>
                <span>{mode.native_surface}</span>
              </div>
              <span className={`mode-status ${statusClass(mode.status)}`}>{mode.status}</span>
            </article>
          );
        })}
      </div>
    </div>
  );
}
