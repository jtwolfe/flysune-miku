import { create } from "zustand";
import { AVOCADO_GROW_PARAGRAPH, TwoSwarmEngine } from "./engine/pipeline";
import type { SpeakTrace } from "./engine/types";

export type LabPhase = "boot" | "ready" | "running" | "error";
export type RevealStage = "idle" | "context" | "kc" | "vote" | "speak" | "done";

export type SlotRef = { token: number; slot: number };

export { AVOCADO_GROW_PARAGRAPH };

type LabState = {
  text: string;
  phase: LabPhase;
  error: string | null;
  engine: TwoSwarmEngine | null;
  trace: SpeakTrace | null;
  speed: number;
  auditionCrumbs: boolean;
  revealing: SlotRef | null;
  revealStage: RevealStage;
  revealedCount: number;
  selected: SlotRef | null;
  playingFull: boolean;
  skipRequested: boolean;
  paused: boolean;
  setText: (t: string) => void;
  setSpeed: (s: number) => void;
  setAudition: (v: boolean) => void;
  setSelected: (s: SlotRef | null) => void;
  setPlayingFull: (v: boolean) => void;
  setPaused: (v: boolean) => void;
  boot: () => Promise<void>;
  speak: () => Promise<void>;
  skip: () => void;
};

let enginePromise: Promise<TwoSwarmEngine> | null = null;

function getEngine() {
  if (!enginePromise) enginePromise = TwoSwarmEngine.load();
  return enginePromise;
}

export const useLab = create<LabState>((set, get) => ({
  text: AVOCADO_GROW_PARAGRAPH,
  phase: "boot",
  error: null,
  engine: null,
  trace: null,
  speed: 2,
  auditionCrumbs: true,
  revealing: null,
  revealStage: "idle",
  revealedCount: 0,
  selected: null,
  playingFull: false,
  skipRequested: false,
  paused: false,

  setText: (t) => set({ text: t }),
  setSpeed: (s) => set({ speed: s }),
  setAudition: (v) => set({ auditionCrumbs: v }),
  setSelected: (s) => set({ selected: s }),
  setPlayingFull: (v) => set({ playingFull: v }),
  setPaused: (v) => set({ paused: v }),
  skip: () => set({ skipRequested: true, paused: false }),

  boot: async () => {
    try {
      const engine = await getEngine();
      set({ engine, phase: "ready", error: null });
    } catch (e) {
      set({
        phase: "error",
        error: e instanceof Error ? e.message : "Failed to load fly models",
      });
    }
  },

  speak: async () => {
    const { text } = get();
    set({
      phase: "running",
      error: null,
      trace: null,
      revealing: null,
      revealStage: "idle",
      revealedCount: 0,
      selected: null,
      skipRequested: false,
      paused: false,
      playingFull: false,
    });
    await new Promise((r) => setTimeout(r, 40));
    try {
      const engine = await getEngine();
      const trace = engine.speak(text);
      set({ engine, trace, phase: "ready" });
    } catch (e) {
      set({
        phase: "error",
        error: e instanceof Error ? e.message : "Speak failed",
      });
    }
  },
}));
