import type { ReactNode } from "react";
import { Activity, Archive, GitBranch, Library, SlidersHorizontal, Upload } from "lucide-react";

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
        <section className={`operator-surface sound-bench ${selected ? "has-selection" : "idle"}`}>
          <div className="surface-head">{currentHeader}</div>
          <InstrumentFlowStrip nodes={flowNodes} />
          <div className="sound-bench-core">
            <div className="sound-bench-lane">{currentSound}</div>
            <aside className="source-rail material-bay" aria-label="Material bay">
              <div className="rail-head">
                <div>
                  <span className="eyebrow">Material Bay</span>
                  <strong>{sourceCount} usable sources</strong>
                </div>
                {importControl}
              </div>
              {materialBay}
            </aside>
          </div>
          <div className="control-bank instrument-control-bank" aria-label="Gesture controls">
            {gestureRack}
            {nextActions}
          </div>
        </section>

        <section className="gesture-workbench tune-bank-zone" aria-label="Tune bank">
          <div className="zone-label">
            <SlidersHorizontal size={16} />
            <span>Tune Bank</span>
          </div>
          {tuneBank}
        </section>

        <section className="result-rail take-field-zone" aria-label="Take field">
          <div className="rail-head">
            <div>
              <span className="eyebrow">Take Field</span>
              <strong>{branchCount} branches</strong>
            </div>
            <Activity size={19} />
          </div>
          <div className="take-field-grid">
            <div className="take-lane-stack">
              {takeQueue}
              {pendingTakes}
              {forkPanel}
            </div>
            <div className="branch-stack-zone">
              <div className="zone-label">
                <GitBranch size={16} />
                <span>Branches</span>
              </div>
              {branchList}
              {branchDetail}
            </div>
          </div>
        </section>

        <section className="memory-zone" aria-label="Memory and evidence">
          <div className="zone-label">
            <Library size={16} />
            <span>Memory</span>
          </div>
          {sessionMemory}
        </section>

        <section className="evidence-zone" aria-label="Evidence dock">
          <div className="zone-label">
            <Archive size={16} />
            <span>Inspect / Evidence</span>
          </div>
          {evidenceDock}
          {utilityDock}
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
