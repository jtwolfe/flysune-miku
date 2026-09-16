import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { floatToWav, playSamples, waitForSource } from "@/lib/engine/audio";
import { AVOCADO_GROW_PARAGRAPH, SAMPLE_RATE } from "@/lib/engine/pipeline";
import type { SpeakTrace, TokenTrace } from "@/lib/engine/types";
import { useLab, type SlotRef } from "@/lib/store";
import { Download, Pause, Play, SkipForward, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { KcGrid } from "./kc-grid";
import { LetterWindow } from "./letter-window";
import { SpeakerPanel } from "./speaker-panel";
import { StageRail } from "./stage-rail";
import { Timeline } from "./timeline";
import { VoteChoir } from "./vote-choir";
import { Waveform } from "./waveform";

const SPEEDS = [0.25, 0.5, 1, 2, 4];
const STAGE_MS = { context: 220, kc: 480, vote: 560, speak: 280 } as const;
const STAGES = ["context", "kc", "vote", "speak"] as const;

function flatten(trace: SpeakTrace | null) {
  if (!trace) return [];
  const out: SlotRef[] = [];
  trace.tokens.forEach((_, ti) => {
    trace.tokens[ti]!.slots.forEach((_, si) => out.push({ token: ti, slot: si }));
  });
  return out;
}

function slotAt(trace: SpeakTrace, ref: SlotRef) {
  return trace.tokens[ref.token]?.slots[ref.slot] ?? null;
}

function tokenAt(trace: SpeakTrace, ref: SlotRef): TokenTrace | null {
  return trace.tokens[ref.token] ?? null;
}

function slotAtSample(trace: SpeakTrace, sample: number): SlotRef | null {
  let best: SlotRef | null = null;
  trace.tokens.forEach((tok, ti) => {
    tok.slots.forEach((slot, si) => {
      if (slot.startSample <= sample) best = { token: ti, slot: si };
    });
  });
  return best;
}

export function LabApp() {
  const boot = useLab((s) => s.boot);
  const speak = useLab((s) => s.speak);
  const skip = useLab((s) => s.skip);
  const phase = useLab((s) => s.phase);
  const error = useLab((s) => s.error);
  const text = useLab((s) => s.text);
  const setText = useLab((s) => s.setText);
  const trace = useLab((s) => s.trace);
  const speed = useLab((s) => s.speed);
  const setSpeed = useLab((s) => s.setSpeed);
  const audition = useLab((s) => s.auditionCrumbs);
  const setAudition = useLab((s) => s.setAudition);
  const revealing = useLab((s) => s.revealing);
  const revealStage = useLab((s) => s.revealStage);
  const revealedCount = useLab((s) => s.revealedCount);
  const selected = useLab((s) => s.selected);
  const setSelected = useLab((s) => s.setSelected);
  const playingFull = useLab((s) => s.playingFull);
  const setPlayingFull = useLab((s) => s.setPlayingFull);
  const paused = useLab((s) => s.paused);
  const setPaused = useLab((s) => s.setPaused);
  const wiring = useLab((s) => s.engine?.picker.manifest.wiring.source ?? "—");
  const fullSrc = useRef<AudioBufferSourceNode | null>(null);
  const crumbSrc = useRef<AudioBufferSourceNode | null>(null);
  const skipRef = useRef(false);
  const pauseRef = useRef(false);
  const [listenProgress, setListenProgress] = useState(0);
  const [listenOrigin, setListenOrigin] = useState(0);

  useEffect(() => {
    void boot();
  }, [boot]);

  skipRef.current = useLab.getState().skipRequested;
  pauseRef.current = paused;

  const slots = useMemo(() => flatten(trace), [trace]);

  useEffect(() => {
    if (!trace || slots.length === 0) return;
    let cancelled = false;
    skipRef.current = false;

    const wait = async (ms: number) => {
      let remaining = ms;
      let last = performance.now();
      while (remaining > 0) {
        if (cancelled || skipRef.current) return;
        if (pauseRef.current) {
          await new Promise((r) => setTimeout(r, 50));
          last = performance.now();
          continue;
        }
        await new Promise((r) => setTimeout(r, 16));
        const now = performance.now();
        remaining -= now - last;
        last = now;
      }
    };

    const stopCrumb = () => {
      try {
        crumbSrc.current?.stop();
      } catch {
        /* already stopped */
      }
      crumbSrc.current = null;
    };

    const run = async () => {
      for (let i = 0; i < slots.length; i++) {
        if (cancelled || skipRef.current) break;
        const ref = slots[i]!;
        useLab.setState({ revealing: ref, selected: ref, revealedCount: i });
        for (const st of STAGES) {
          if (cancelled || skipRef.current) break;
          useLab.setState({ revealStage: st });
          await wait(STAGE_MS[st] / Math.max(0.25, useLab.getState().speed));
        }
        if (cancelled) break;
        useLab.setState({ revealedCount: i + 1, revealStage: "speak" });
        const sl = slotAt(trace, ref);
        if (sl && useLab.getState().auditionCrumbs && !skipRef.current) {
          try {
            stopCrumb();
            const src = await playSamples(sl.crumb);
            crumbSrc.current = src;
            await Promise.race([
              waitForSource(src),
              wait((sl.crumb.length / SAMPLE_RATE) * 1000 + 80),
            ]);
          } catch {
            /* autoplay */
          }
        }
      }
      stopCrumb();
      if (!cancelled) {
        useLab.setState({
          revealedCount: slots.length,
          revealStage: "done",
          revealing: slots[slots.length - 1] ?? null,
          skipRequested: false,
          paused: false,
        });
      }
    };
    void run();
    return () => {
      cancelled = true;
      stopCrumb();
    };
  }, [trace, slots]);

  useEffect(() => {
    if (!playingFull || !trace) {
      setListenProgress(0);
      return;
    }
    const start = performance.now();
    const origin = listenOrigin;
    let raf = 0;
    const tick = () => {
      const elapsed = (performance.now() - start) / 1000;
      const sample = origin + elapsed * trace.sampleRate;
      const p = Math.min(1, sample / Math.max(1, trace.audio.length));
      setListenProgress(p);
      const ref = slotAtSample(trace, sample);
      if (ref) useLab.setState({ selected: ref, revealing: ref });
      if (useLab.getState().playingFull && p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playingFull, trace, listenOrigin]);

  useEffect(() => {
    return () => {
      try {
        fullSrc.current?.stop();
      } catch {
        /* already stopped */
      }
      fullSrc.current = null;
    };
  }, [trace]);

  const liveRef = selected ?? revealing;
  const liveSlot = trace && liveRef ? slotAt(trace, liveRef) : null;
  const liveToken = trace && liveRef ? tokenAt(trace, liveRef) : null;
  const matchPct = trace && trace.nTotal ? Math.round((100 * trace.nMatch) / trace.nTotal) : 0;
  const waveProgress = playingFull
    ? listenProgress
    : liveSlot && trace && trace.audio.length
      ? liveSlot.startSample / trace.audio.length
      : revealedCount / Math.max(1, slots.length);
  const revealingNow = revealStage !== "idle" && revealStage !== "done" && !!trace;
  const busy = phase === "boot" || phase === "running";
  const nCmu = trace ? trace.tokens.filter((t) => t.slotCountSource === "cmudict").length : 0;
  const nSimple = trace ? trace.tokens.filter((t) => t.slotCountSource === "simple").length : 0;

  function stopFull() {
    try {
      fullSrc.current?.stop();
    } catch {
      /* already stopped */
    }
    fullSrc.current = null;
  }

  function stopCrumb() {
    try {
      crumbSrc.current?.stop();
    } catch {
      /* already stopped */
    }
    crumbSrc.current = null;
  }

  async function startListen(offsetSamples = 0) {
    if (!trace) return;
    stopCrumb();
    stopFull();
    skipRef.current = true;
    skip();
    setListenOrigin(offsetSamples);
    const src = await playSamples(trace.audio, trace.sampleRate, offsetSamples);
    fullSrc.current = src;
    setPlayingFull(true);
    src.onended = () => {
      if (fullSrc.current === src) {
        setPlayingFull(false);
        fullSrc.current = null;
      }
    };
  }

  async function togglePlay() {
    if (!trace) return;
    if (playingFull) {
      stopFull();
      setPlayingFull(false);
      return;
    }
    await startListen(0);
  }

  function downloadWav() {
    if (!trace) return;
    const blob = floatToWav(trace.audio, trace.sampleRate);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "flysune.wav";
    a.click();
    URL.revokeObjectURL(url);
  }

  function inspect(ref: SlotRef) {
    setSelected(ref);
    if (revealingNow) {
      pauseRef.current = true;
      setPaused(true);
    }
  }

  function seekWave(t: number) {
    if (!trace) return;
    const sample = t * trace.audio.length;
    const ref = slotAtSample(trace, sample);
    if (ref) inspect(ref);
    if (playingFull) void startListen(sample);
  }

  return (
    <div className="mx-auto flex min-h-dvh max-w-7xl flex-col gap-4 px-4 py-5 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-2xs tracking-[0.18em] text-muted uppercase">Cursed TTS · v0.3.1</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight text-balance text-fg sm:text-3xl">
            Flysune Miku
          </h1>
          <p className="mt-1 max-w-xl text-sm text-pretty text-muted">
            Recognition flies pick phonemes. Speaking flies speak them. Speakers never see Kenyon cells.
          </p>
        </div>
        <img
          src="/assets/flysune-miku.jpg"
          alt="Flysune Miku project mascot"
          className="size-16 rounded-lg object-cover shadow-[var(--shadow-border)] sm:size-20"
        />
      </header>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <aside className="flex flex-col gap-3">
          <label className="text-2xs font-medium tracking-wide text-muted uppercase" htmlFor="utterance">
            Utterance
          </label>
          <Textarea
            id="utterance"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            spellCheck={false}
          />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void speak()} disabled={busy || !text.trim()}>
              {phase === "running" ? "Speaking…" : "Speak"}
            </Button>
            <Button variant="secondary" onClick={() => void togglePlay()} disabled={!trace}>
              {playingFull ? <Pause className="size-4" /> : <Play className="size-4" />}
              {playingFull ? "Stop" : "Listen"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                if (revealingNow && !paused) {
                  pauseRef.current = true;
                  setPaused(true);
                } else if (revealingNow && paused) {
                  pauseRef.current = false;
                  setPaused(false);
                }
              }}
              disabled={!revealingNow}
            >
              {paused ? <Play className="size-4" /> : <Pause className="size-4" />}
              {paused ? "Resume" : "Pause"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                skipRef.current = true;
                stopCrumb();
                skip();
              }}
              disabled={!trace || revealStage === "done"}
            >
              <SkipForward className="size-4" />
              Skip
            </Button>
            <Button variant="outline" onClick={downloadWav} disabled={!trace}>
              <Download className="size-4" />
              WAV
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => setText("cat bat dog")}>
              cat bat dog
            </Button>
            <Button size="sm" variant="outline" onClick={() => setText("Hi me")}>
              Hi me
            </Button>
            <Button size="sm" variant="outline" onClick={() => setText(AVOCADO_GROW_PARAGRAPH)}>
              Avocado paragraph
            </Button>
          </div>
          <div className="flex items-center justify-between gap-3 rounded-xl bg-surface px-3 py-2 shadow-[var(--shadow-border)]">
            <span className="text-xs text-muted">Slow-mo</span>
            <div className="flex gap-1">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSpeed(s)}
                  className={
                    s === speed
                      ? "min-h-11 rounded-md bg-kc/20 px-2 font-mono text-xs text-kc"
                      : "min-h-11 rounded-md px-2 font-mono text-xs text-muted hover:text-fg"
                  }
                >
                  {s}×
                </button>
              ))}
            </div>
          </div>
          <label className="flex min-h-11 items-center justify-between gap-3 rounded-xl bg-surface px-3 shadow-[var(--shadow-border)]">
            <span className="flex items-center gap-2 text-xs text-muted">
              <Volume2 className="size-3.5" />
              Audition crumbs
            </span>
            <Switch checked={audition} onCheckedChange={setAudition} />
          </label>
          <Honesty
            traceReady={!!trace}
            matchPct={matchPct}
            nMatch={trace?.nMatch ?? 0}
            nTotal={trace?.nTotal ?? 0}
            nCmu={nCmu}
            nSimple={nSimple}
            wiring={wiring}
          />
          {error && (
            <p className="text-sm text-danger" role="alert">
              {error}
            </p>
          )}
          {phase === "boot" && <p className="text-sm text-muted">Loading picker and speaker weights…</p>}
          {phase === "running" && <p className="text-sm text-kc">Flies are classifying…</p>}
        </aside>

        <section className="flex min-w-0 flex-col gap-3">
          <StageRail stage={revealStage} />
          {trace && (
            <p className="font-mono text-xs tabular-nums text-muted" aria-live="polite">
              {liveToken?.word ?? "—"} · slot {liveSlot ? `${liveSlot.index + 1}/${liveSlot.nPhonemes}` : "—"} ·{" "}
              {revealedCount}/{slots.length} revealed · {(trace.audio.length / trace.sampleRate).toFixed(2)} s
            </p>
          )}
          <Timeline
            trace={trace}
            revealedCount={revealedCount}
            selected={liveRef}
            onSelect={inspect}
          />
          <Waveform trace={trace} progress={waveProgress} onSeek={seekWave} />
          <LetterWindow slot={liveSlot} token={liveToken} />
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
              <KcGrid
                slot={liveSlot}
                progress={
                  revealStage === "context" ? 0.12 : revealStage === "kc" ? 0.7 : 1
                }
              />
            </div>
            <div className="flex min-h-64 flex-col rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
              <VoteChoir
                slot={liveSlot}
                progress={
                  revealStage === "vote" || revealStage === "speak" || revealStage === "done"
                    ? 1
                    : revealStage === "kc"
                      ? 0.2
                      : 0.08
                }
              />
            </div>
          </div>
          <SpeakerPanel slot={liveSlot} />
        </section>
      </div>

      <footer className="mt-auto border-t border-border pt-4 text-2xs text-subtle">
        Project mascot only. Not affiliated with Crypton Future Media. Marian crumbs: Kanabun / MARIAN
        ILUSTRADO. Hemibrain: Scheffer et al. 2020, CC-BY. Architecture: picker G2P → phoneme ids →
        speakers.
      </footer>
    </div>
  );
}

function Honesty({
  traceReady,
  matchPct,
  nMatch,
  nTotal,
  nCmu,
  nSimple,
  wiring,
}: {
  traceReady: boolean;
  matchPct: number;
  nMatch: number;
  nTotal: number;
  nCmu: number;
  nSimple: number;
  wiring: string;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-xl bg-surface p-4 shadow-[var(--shadow-border)]">
      <h3 className="text-2xs font-medium tracking-wide text-muted uppercase">Honesty strip</h3>
      <ul className="flex flex-col gap-1.5 text-xs text-muted">
        <li className="flex items-center justify-between gap-2">
          Slot count
          <Badge>
            {traceReady ? `${nCmu} CMUdict · ${nSimple} simple` : "CMUdict / simple G2P"}
          </Badge>
        </li>
        <li className="flex items-center justify-between gap-2">
          Picker vs reference
          <Badge tone={traceReady && matchPct < 80 ? "warn" : "ok"}>
            {traceReady ? `${nMatch}/${nTotal} · ${matchPct}%` : "—"}
          </Badge>
        </li>
        <li className="flex items-center justify-between gap-2">
          Speakers see
          <Badge tone="kc">phoneme id only</Badge>
        </li>
        <li className="flex items-center justify-between gap-2">
          PN→KC wiring
          <Badge>{wiring}</Badge>
        </li>
      </ul>
    </div>
  );
}
