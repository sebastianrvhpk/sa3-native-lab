import type { ReactNode } from "react";
import { Activity, Archive, Box, GitBranch, Library, SlidersHorizontal, Upload } from "lucide-react";

import type { InstrumentFlowNode } from "./instrumentFrameModel";

interface SoundInstrumentSurfaceProps {
  selected: boolean;
  brand: ReactNode;
  question: string;
  loopPrompt: string;
  settings: ReactNode;
  currentHeader: ReactNode;
  currentSound: ReactNode;
  flowNodes: readonly InstrumentFlowNode[];
  sourceCount: number;
  branchCount: number;
  importControl: ReactNode;
  materialBay: ReactNode;
  gestureRack: ReactNode;
  nextActions: ReactNode;
  tuneBank: ReactNode;
  takeQueue: ReactNode;
  pendingTakes: ReactNode;
  forkPanel: ReactNode;
  branchList: ReactNode;
  branchDetail: ReactNode;
  sessionMemory: ReactNode;
  evidenceDock: ReactNode;
  utilityDock: ReactNode;
}

export function SoundInstrumentSurface({
  selected,
  brand,
  question,
  loopPrompt,
  settings,
  currentHeader,
  currentSound,
  flowNodes,
  sourceCount,
  branchCount,
  importControl,
  materialBay,
  gestureRack,
  nextActions,
  tuneBank,
  takeQueue,
  pendingTakes,
  forkPanel,
  branchList,
  branchDetail,
  sessionMemory,
  evidenceDock,
  utilityDock,
}: SoundInstrumentSurfaceProps) {
  return (
    <main className="app-shell instrument-shell">
      <header className="top-strip">
        {brand}
        <div className="instrument-question">
          <span>{question}</span>
          <small>{loopPrompt}</small>
        </div>
        {settings}
      </header>

      <section className="instrument-board" aria-label="SA3 Native Lab sound instrument">
        <section className={`operator-surface sound-stage ${selected ? "has-selection" : "idle"}`}>
          <div className="stage-header">
            <div className="surface-head">{currentHeader}</div>
            <InstrumentFlowStrip nodes={flowNodes} />
          </div>

          <div className="stage-main">
            <div className="sound-stage-player">{currentSound}</div>
            <aside className="gesture-workbench gesture-console" aria-label="Prompt and gesture">
              <div className="zone-label">
                <SlidersHorizontal size={16} />
                <span>Gesture / Tune</span>
              </div>
              {gestureRack}
              {tuneBank}
            </aside>
          </div>

          <div className="take-lane-panel" aria-label="Take lane">
            <div className="rail-head">
              <div>
                <span className="eyebrow">Take Lane</span>
                <strong>listen next</strong>
              </div>
              <Activity size={19} />
            </div>
            <div className="take-lane-grid">
              {takeQueue}
              {pendingTakes}
              {forkPanel}
            </div>
          </div>
        </section>

        <section className="instrument-trays" aria-label="Secondary instrument trays">
          <details className="instrument-tray action-tray">
            <summary>
              <span>
                <Activity size={16} />
                Do next
              </span>
              <strong>optional moves</strong>
            </summary>
            <div className="tray-body">{nextActions}</div>
          </details>

          <details className="instrument-tray material-tray">
            <summary>
              <span>
                <Box size={16} />
                Material
              </span>
              <strong>{sourceCount} sources</strong>
            </summary>
            <div className="tray-body">
              <div className="tray-actions">{importControl}</div>
              {materialBay}
            </div>
          </details>

          <details className="instrument-tray branch-tray">
            <summary>
              <span>
                <GitBranch size={16} />
                Branches
              </span>
              <strong>{branchCount} paths</strong>
            </summary>
            <div className="tray-body branch-tray-body">
              {branchList}
              {branchDetail}
            </div>
          </details>

          <details className="instrument-tray memory-tray">
            <summary>
              <span>
                <Library size={16} />
                Memory
              </span>
              <strong>remembered material</strong>
            </summary>
            <div className="tray-body">{sessionMemory}</div>
          </details>

          <details className="instrument-tray inspect-tray">
            <summary>
              <span>
                <Archive size={16} />
                Inspect
              </span>
              <strong>runtime and provenance</strong>
            </summary>
            <div className="tray-body">
              {evidenceDock}
              {utilityDock}
            </div>
          </details>
        </section>
      </section>
    </main>
  );
}

function InstrumentFlowStrip({ nodes }: { nodes: readonly InstrumentFlowNode[] }) {
  return (
    <div className="instrument-flow-strip" aria-label="Instrument loop">
      {nodes.map((node, index) => (
        <div key={node.id} className={`instrument-flow-node ${node.tone}`}>
          {index ? <span className="instrument-flow-route" aria-hidden="true" /> : null}
          <span>{node.label}</span>
          <strong>{node.value}</strong>
        </div>
      ))}
    </div>
  );
}

export function ImportAudioButton({
  children,
  onFile,
}: {
  children?: ReactNode;
  onFile: (file: File) => void;
}) {
  return (
    <label className="icon-button" title="Import audio">
      {children ?? <Upload size={18} />}
      <input
        type="file"
        accept="audio/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFile(file);
          event.currentTarget.value = "";
        }}
      />
    </label>
  );
}
