import { Archive, CircleAlert, GitFork, Info, RotateCcw, SlidersHorizontal, Wand2 } from "lucide-react";

import type { ProductNextAction } from "./nextActionModel";

export function NextActionsPanel({
  actions,
  onAction,
}: {
  actions: readonly ProductNextAction[];
  onAction: (action: ProductNextAction) => void;
}) {
  if (!actions.length) return null;
  return (
    <section className="next-actions-panel" aria-label="Do next">
      <div className="next-actions-head">
        <span className="eyebrow">Next</span>
        <strong>Do next</strong>
      </div>
      <div className="next-actions-list">
        {actions.slice(0, 6).map((action) => (
          <button
            key={action.id}
            type="button"
            data-action-kind={action.kind}
            disabled={!action.available}
            title={action.disabledReason ?? action.description}
            onClick={() => onAction(action)}
          >
            <NextActionIcon action={action} />
            <span>{action.label}</span>
            <small>{action.available ? action.description : action.disabledReason}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function NextActionIcon({ action }: { action: ProductNextAction }) {
  if (!action.available) return <CircleAlert size={16} />;
  if (action.kind === "remember") return <Archive size={16} />;
  if (action.kind === "branch") return <GitFork size={16} />;
  if (action.kind === "retry") return <RotateCcw size={16} />;
  if (action.kind === "inspect") return <Info size={16} />;
  if (action.kind === "tune") return <SlidersHorizontal size={16} />;
  return <Wand2 size={16} />;
}
