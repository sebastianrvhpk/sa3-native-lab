import type { ReactNode } from "react";
import { CircleAlert, SlidersHorizontal } from "lucide-react";

import type { GestureActionDescriptor } from "./gestureActionDescriptor";
import type { GestureOption } from "./gestureModel";

interface TuneDrawerProps {
  gesture: GestureOption;
  children: ReactNode;
  inspect?: ReactNode;
  action?: ReactNode;
  actionDescriptor?: GestureActionDescriptor;
}

export function TuneDrawer({ gesture, children, inspect, action, actionDescriptor }: TuneDrawerProps) {
  return (
    <section className={`tune-drawer ${gesture.available ? "available" : "blocked"}`} aria-label={`${gesture.label} tune`}>
      <div className="tune-head">
        <div>
          <span className="eyebrow">Active Gesture</span>
          <h2>{gesture.label}</h2>
          <p>{gesture.shortIntent}</p>
        </div>
        <div className="tune-head-controls">
          <span className={`gesture-state ${gesture.available ? "ready" : "blocked"}`}>
            {gesture.available ? "ready" : "needs source"}
          </span>
          {action ? <div className="gesture-action tune-primary-action">{action}</div> : null}
        </div>
      </div>
      {!gesture.available ? (
        <div className="quiet-panel compact gesture-blocked">
          <CircleAlert size={16} />
          {gesture.disabledReason}
        </div>
      ) : null}
      {actionDescriptor ? <GestureActionSummary descriptor={actionDescriptor} /> : null}
      <details className="tune-details" open>
        <summary>
          <SlidersHorizontal size={15} />
          Tune
        </summary>
        <div className="tune-body">{children}</div>
      </details>
      {inspect ? (
        <details className="contract-details">
          <summary>Inspect gesture</summary>
          {inspect}
        </details>
      ) : null}
    </section>
  );
}

function GestureActionSummary({ descriptor }: { descriptor: GestureActionDescriptor }) {
  return (
    <div className={`gesture-action-summary ${descriptor.ready ? "ready" : "blocked"}`}>
      <p>{descriptor.disabledReason ?? descriptor.intentCopy}</p>
      {descriptor.sourceRequirements.length ? (
        <div className="gesture-requirements" aria-label="Gesture source requirements">
          {descriptor.sourceRequirements.map((requirement) => (
            <span key={`${requirement.label}:${requirement.detail}`} className={requirement.status} title={requirement.detail}>
              {requirement.label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
