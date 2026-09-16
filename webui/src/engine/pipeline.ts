import { buildLetterContext, slotGeometry } from "./encode";
import { getPhonemes, loadLexicon } from "./lexicon";
import { MoreFlyPicker } from "./picker";
import { concatCrumbs, SAMPLE_RATE, SpeakerSwarmJs } from "./speaker";
import type { Phoneme, SlotTrace, SpeakTrace, TokenTrace } from "./types";

export { SAMPLE_RATE };

export const WORD_GAP_S = 0.15;
export const COMMA_PAUSE_S = 0.3;
export const PERIOD_PAUSE_S = 0.55;
const PUNCT: Record<string, number> = { ",": COMMA_PAUSE_S, ".": PERIOD_PAUSE_S };

export const AVOCADO_GROW_PARAGRAPH =
  "Avocados grow on trees. The trees are tall, and the fruit is green. When avocados are ripe, people pick them. The fruit has a large seed inside.";

export type Token = { word: string; pauseS: number };

export function tokenizeSpokenText(text: string): Token[] {
  const items: { word: string; pause: number }[] = [];
  for (const raw of text.trim().split(/\s+/)) {
    if (!raw) continue;
    let pause = WORD_GAP_S;
    let core = raw;
    while (core && core[core.length - 1]! in PUNCT) {
      pause = Math.max(pause, PUNCT[core[core.length - 1]!]!);
      core = core.slice(0, -1);
    }
    while (core && core[0]! in PUNCT) {
      pause = Math.max(pause, PUNCT[core[0]!]!);
      core = core.slice(1);
    }
    const word = [...core].filter((c) => /[a-zA-Z]/.test(c)).join("");
    if (!word) {
      if (items.length && pause > items[items.length - 1]!.pause) {
        items[items.length - 1]!.pause = pause;
      }
      continue;
    }
    items.push({ word, pause });
  }
  if (items.length && items[items.length - 1]!.pause === WORD_GAP_S) {
    items[items.length - 1]!.pause = 0;
  }
  return items.map((it) => ({ word: it.word, pauseS: it.pause }));
}

function silence(seconds: number): Float32Array {
  return new Float32Array(Math.max(0, Math.round(seconds * SAMPLE_RATE)));
}

function concatParts(parts: Float32Array[]): Float32Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const audio = new Float32Array(total);
  let o = 0;
  for (const p of parts) {
    audio.set(p, o);
    o += p.length;
  }
  return audio;
}

export class TwoSwarmEngine {
  picker: MoreFlyPicker;
  speakers: SpeakerSwarmJs;

  constructor(picker: MoreFlyPicker, speakers: SpeakerSwarmJs) {
    this.picker = picker;
    this.speakers = speakers;
  }

  static async load(): Promise<TwoSwarmEngine> {
    const [picker, speakers] = await Promise.all([
      MoreFlyPicker.load(),
      SpeakerSwarmJs.load(),
      loadLexicon(),
    ]);
    return new TwoSwarmEngine(picker, speakers);
  }

  speakWord(wordRaw: string): TokenTrace {
    const word = wordRaw.toLowerCase();
    const { phones: ref, source, known } = getPhonemes(word);
    const nPhonemes = ref.length;
    const slots: SlotTrace[] = [];
    const pickerPhones: Phoneme[] = [];
    let prev: Phoneme | null = null;
    const contextSize = this.picker.manifest.contextSize;

    for (let pIdx = 0; pIdx < nPhonemes; pIdx++) {
      const { letterPos, phonemePos } = slotGeometry(word, pIdx, nPhonemes);
      const letterContext = buildLetterContext(word, letterPos, contextSize);
      const pred = this.picker.predict(letterContext, phonemePos, nPhonemes, prev);
      const position = nPhonemes > 1 ? pIdx / (nPhonemes - 1) : 0.5;
      const crumb = this.speakers.synthesizePhoneme(pred.winner, prev, position);
      const params = this.speakers.getParams(pred.winner, prev, position);
      slots.push({
        index: pIdx,
        letterContext,
        letterPos,
        phonemePos,
        nPhonemes,
        previousPhone: prev,
        referencePhone: ref[pIdx]!,
        pickerPhone: pred.winner,
        confidence: pred.confidence,
        nKcActive: pred.nKcActive,
        nKc: this.picker.manifest.nKc,
        kcActive: pred.kcActive,
        votes: pred.votes,
        speaker: { ...params, phoneme: pred.winner, durationMs: params.durationMs },
        crumb,
        startSample: 0,
      });
      pickerPhones.push(pred.winner);
      prev = pred.winner;
    }

    const { starts } = concatCrumbs(
      slots.map((s) => s.crumb),
      this.speakers.crossfadeMs,
    );
    for (let i = 0; i < slots.length; i++) slots[i]!.startSample = starts[i] ?? 0;

    const nMatch = slots.reduce(
      (n, s) => n + (s.pickerPhone === s.referencePhone ? 1 : 0),
      0,
    );
    return {
      word,
      pauseS: 0,
      isKnown: known,
      slotCountSource: source,
      referencePhonemes: ref,
      pickerPhonemes: pickerPhones,
      nMatch,
      slots,
    };
  }

  speak(text: string): SpeakTrace {
    const tokensIn = tokenizeSpokenText(text);
    const tokens: TokenTrace[] = [];
    const parts: Float32Array[] = [];
    let nMatch = 0;
    let nTotal = 0;
    let cursor = 0;

    for (const tok of tokensIn) {
      const traced = this.speakWord(tok.word);
      traced.pauseS = tok.pauseS;
      const wordAudio = concatCrumbs(
        traced.slots.map((s) => s.crumb),
        this.speakers.crossfadeMs,
      );
      for (const slot of traced.slots) slot.startSample += cursor;
      tokens.push(traced);
      nMatch += traced.nMatch;
      nTotal += traced.slots.length;
      parts.push(wordAudio.audio);
      cursor += wordAudio.audio.length;
      if (tok.pauseS > 0) {
        const gap = silence(tok.pauseS);
        parts.push(gap);
        cursor += gap.length;
      }
    }

    return {
      text,
      tokens,
      sampleRate: SAMPLE_RATE,
      audio: concatParts(parts),
      nMatch,
      nTotal,
      architecture: "picker G2P → phoneme ids → Marian speaker flies (speakers never see KC)",
    };
  }
}