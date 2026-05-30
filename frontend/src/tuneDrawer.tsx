import type { ReactNode } from "react";
import { CircleAlert, SlidersHorizontal } from "lucide-react";

import type { GestureOption } from "./gestureModel";

interface TuneDrawerProps {
  gesture: GestureOption;
  children: ReactNode;
  inspect?: ReactNode;
  action?: ReactNode;
}

export function TuneDrawer({ gesture, children, inspect, action }: TuneDrawerProps) {
  return (
    <section className={`tune-drawer ${gesture.available ? "available" : "blocked"}`} aria-label={`${gesture.label} tune`}>
      <div className="tune-head">
        <div>
          <span className="eyebrow">Active Gesture</span>
          <h2>{gesture.label}</h2>
          <p>{gesture.shortIntent}</p>
        </div>
        <span className={`gesture-state ${gesture.available ? "ready" : "blocked"}`}>
          {gesture.available ? "ready" : "needs source"}
        </span>
      </div>
      {!gesture.available ? (
        <div className="quiet-panel compact gesture-blocked">
          <CircleAlert size={16} />
          {gesture.disabledReason}
        </div>
      ) : null}
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
      {action ? <div className="gesture-action">{action}</div> : null}
    </section>
  );
}
