"use client";

/**
 * Upload a picture, choose a region, watch the edit happen.
 *
 * The flow exists to make the chooser reachable. ADR-0003 measured that letting the user
 * pick from five candidates takes grounding from 0.516 to 0.832 on held-out RefCOCOg, and
 * shipped the API for it — `POST /v1/masks` — with nothing on the other end. Every
 * candidate below has been computed and discarded on every edit since.
 *
 * **Grounding is a separate step from editing, deliberately.** It is cheap and reversible
 * where an edit is neither, so the region is settled before any model time is spent, and
 * the chosen mask travels with the job rather than being grounded a second time.
 *
 * **The chooser is not in front of every edit.** ADR-0003 rejected always-asking: on the
 * 45% of phrases where the detector scores 0.95 against 0.07 it is friction for nothing.
 * When the gateway says the answer is not ambiguous this shows the region it found and
 * gets on with it, with the alternatives one tap away rather than zero.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { GetToken } from "@/lib/api";
import {
  ApiError,
  GATEWAY_URL,
  createJob,
  getJob,
  groundPhrase,
  groundPoints,
  streamJob,
  uploadImage,
  type Candidate,
  type MaskPayload,
  type PointPrompt,
  type UploadedImage,
} from "@/lib/api";
import {
  OPERATIONS,
  PHRASE,
  buildJob,
  groundable,
  ready,
  specFor,
  type Operation,
  type Region,
} from "@/lib/edit-request";
import { EMPTY_HISTORY, hasMask, type MaskHistory } from "@/lib/mask-history";
import { maskFromStrokes, maskSize } from "@/lib/strokes";
import { ORIGINAL, describe as describeEdit, record, type Version } from "@/lib/versions";
import { BeforeAfter } from "./BeforeAfter";
import { BrushCanvas } from "./BrushCanvas";
import { CandidatePicker, REJECTED } from "./CandidatePicker";
import { ImageDrop } from "./ImageDrop";
import { MaskPreview } from "./MaskPreview";
import { StepTimeline, type Step } from "./StepTimeline";
import { VersionStrip } from "./VersionStrip";

type Phase = "idle" | "uploading" | "grounding" | "running" | "done";

/**
 * Describing the region or drawing it.
 *
 * Two modes rather than one clever one. A phrase is faster when it works, and ADR-0003
 * measured that it works about five times in six once the user can choose among
 * candidates; the brush is what the remaining sixth needs, along with anything the
 * detector cannot ground at all.
 */
type Mode = "describe" | "draw";

interface Progress {
  readonly state: string;
  readonly progress: number;
  readonly detail: string;
}

export function Workspace({ getToken }: { getToken: GetToken }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const [image, setImage] = useState<UploadedImage | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const [op, setOp] = useState<Operation>("remove");
  const [target, setTarget] = useState("");
  const [content, setContent] = useState("");
  const [colour, setColour] = useState("#2ea043");

  const [mode, setMode] = useState<Mode>("describe");
  const [strokes, setStrokes] = useState<MaskHistory>(EMPTY_HISTORY);
  const [taps, setTaps] = useState<PointPrompt[]>([]);
  const [tapMask, setTapMask] = useState<MaskPayload | null>(null);
  const [tapping, setTapping] = useState(false);
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [ambiguous, setAmbiguous] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState(0);

  const [progress, setProgress] = useState<Progress | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [beforeUrl, setBeforeUrl] = useState<string | null>(null);

  const [versions, setVersions] = useState<readonly Version[]>([]);
  const [current, setCurrent] = useState(0);
  // Object URLs keyed by digest. Held here rather than on `Version` so that stays a plain
  // description of what exists, and so every URL has one owner that revokes it.
  const urls = useRef<Map<string, string>>(new Map());

  const stopStream = useRef<(() => void) | null>(null);
  const spec = specFor(op);

  // Blob URLs are held by the browser until revoked; without this every picture opened
  // in a session stays in memory for the life of the tab.
  useEffect(() => () => stopStream.current?.(), []);
  useEffect(() => () => { if (imageUrl !== null) URL.revokeObjectURL(imageUrl); }, [imageUrl]);
  useEffect(() => {
    const held = urls.current;
    return () => {
      for (const url of held.values()) URL.revokeObjectURL(url);
      held.clear();
    };
  }, []);

  const reportError = (cause: unknown) => {
    if (cause instanceof ApiError) {
      setError(cause.message);
      return;
    }
    // `fetch` rejects with a bare TypeError when the request never reached the server —
    // no status, no body, and the reason only in the console. That message hid a gateway
    // with no CORS middleware for as long as the frontend existed: every call was blocked
    // before it was sent, and the page said "something went wrong".
    if (cause instanceof TypeError) {
      setError(
        `Could not reach the gateway at ${GATEWAY_URL}. It may be down, or not allowing ` +
          "requests from this page. The browser console has the reason.",
      );
      return;
    }
    setError("Something went wrong. Try again.");
  };

  /** Anything that invalidates a region the user already approved. */
  const forgetRegion = useCallback(() => {
    setCandidates(null);
    setAmbiguous(false);
    setExpanded(false);
    setSelected(0);
  }, []);

  /**
   * What the job will act on.
   *
   * Rasterised here rather than kept as pixels, so the history stays strokes right up to
   * the moment of submission — a 50-step history is kilobytes this way and gigabytes the
   * other. Returns `PHRASE` when there is nothing to send, which leaves the worker to
   * ground the words as it always did.
   */
  const regionFor = useCallback((): Region => {
    if (mode === "draw") {
      if (image === null || (!hasMask(strokes) && tapMask === null)) return PHRASE;
      const drawn = maskFromStrokes(
        strokes.strokes,
        maskSize(image.width, image.height),
        tapMask,
      );
      return drawn === null ? PHRASE : { kind: "drawn", mask: drawn };
    }
    if (candidates !== null && selected >= 0) {
      const chosen = candidates[selected]?.mask;
      if (chosen !== undefined) return { kind: "chosen", mask: chosen };
    }
    return PHRASE;
  }, [mode, image, strokes, tapMask, candidates, selected]);

  const resetRegion = useCallback(() => {
    forgetRegion();
    setStrokes(EMPTY_HISTORY);
    setTaps([]);
    setTapMask(null);
  }, [forgetRegion]);

  async function onFile(file: File) {
    resetRegion();
    setError(null);
    setResultUrl(null);
    setSteps([]);
    setPhase("uploading");
    try {
      const uploaded = await uploadImage(getToken, file);
      const url = URL.createObjectURL(file);
      urls.current.set(uploaded.sha256, url);
      setImage(uploaded);
      setImageUrl(url);
      // A new picture is a new session's worth of history, not a branch of the old one.
      setVersions([{ ...uploaded, label: ORIGINAL }]);
      setCurrent(0);
      setPhase("idle");
    } catch (cause) {
      reportError(cause);
      setPhase("idle");
    }
  }

  /** Step back to an earlier result and edit *that* — the branch in the history. */
  function onPickVersion(index: number) {
    const version = versions[index];
    const url = version === undefined ? undefined : urls.current.get(version.sha256);
    if (version === undefined || url === undefined) return;
    resetRegion();
    setError(null);
    setResultUrl(null);
    setSteps([]);
    setPhase("idle");
    setCurrent(index);
    setImage({ ...version, content_type: "image/png", megapixels: 0, url, url_expires_at: 0 });
    setImageUrl(url);
  }

  async function onGround() {
    if (image === null) return;
    setError(null);
    setPhase("grounding");
    try {
      const found = await groundPhrase(getToken, image.sha256, target.trim());
      if (found.candidates.length === 0) {
        // A real answer, not a failure: nothing in this picture matches those words.
        setError(`Nothing here matches “${target.trim()}”. Try describing it differently.`);
        forgetRegion();
      } else {
        setCandidates(found.candidates);
        setAmbiguous(found.ambiguous);
        setExpanded(false);
        setSelected(0);
      }
    } catch (cause) {
      reportError(cause);
    } finally {
      setPhase("idle");
    }
  }

  /**
   * Resolve a tap, with every tap so far.
   *
   * SAM takes the whole prompt each time rather than refining incrementally, so the
   * accumulated list is sent and the answer replaces the previous one. That is what makes
   * a second tap *improve* the selection instead of starting a new one.
   */
  async function onTap(point: PointPrompt) {
    if (image === null) return;
    const wanted = [...taps, point];
    setTaps(wanted);
    setTapping(true);
    setError(null);
    try {
      const found = await groundPoints(getToken, image.sha256, wanted);
      const mask = found.candidates[0]?.mask;
      if (mask === undefined) {
        setError("Nothing to select there. Try tapping the middle of it, or use the brush.");
        setTaps(taps);
      } else {
        setTapMask(mask);
      }
    } catch (cause) {
      reportError(cause);
      setTaps(taps);
    } finally {
      setTapping(false);
    }
  }

  async function onRun() {
    if (image === null) return;
    setError(null);
    setResultUrl(null);
    setProgress(null);
    setSteps([]);
    setBeforeUrl(imageUrl);
    setPhase("running");

    const draft = { op, imageSha256: image.sha256, target, content, colour };
    const label = describeEdit(op, target, content);

    try {
      const job = await createJob(getToken, buildJob(draft, regionFor()));
      stopStream.current = streamJob(getToken, job.id, (event) => {
        setProgress({ state: event.state, progress: event.progress, detail: event.detail });
        // The worker publishes a step per state change, including passes it rolled back.
        setSteps((seen) =>
          seen.length > 0 && seen[seen.length - 1]!.state === event.state && !event.detail
            ? seen
            : [...seen, { state: event.state, detail: event.detail }],
        );
        if (event.terminal) void finish(job.id, label);
      });
    } catch (cause) {
      reportError(cause);
      setPhase("idle");
    }
  }

  const finish = useCallback(
    async (id: string, label: string) => {
      stopStream.current?.();
      try {
        const done = await getJob(getToken, id);
        // The job's own steps are richer than the stream's: they carry timestamps, and any
        // event that arrived while the tab was backgrounded is in them and not in the
        // stream.
        if (done.steps.length > 0) {
          setSteps(done.steps.map((s) => ({ state: s.state, detail: s.detail, at: s.at })));
        }
        if (done.result_sha256 === null) {
          setError(done.error ?? "The edit finished without producing an image.");
          setPhase("idle");
          return;
        }
        const digest = done.result_sha256;
        // The gateway signs the link; no fetch, no object URL, and the browser's own
        // cache does its job.
        const url = done.result_url;
        urls.current.set(digest, url);
        setResultUrl(url);
        setVersions((history) =>
          record(history, current, {
            sha256: digest,
            width: image?.width ?? 0,
            height: image?.height ?? 0,
            label,
          }),
        );
        setCurrent((at) => Math.max(at, 0) + 1);
        setPhase("done");
      } catch (cause) {
        reportError(cause);
        setPhase("idle");
      }
    },
    [getToken, current, image],
  );

  const busy = phase === "uploading" || phase === "grounding" || phase === "running";
  const drawing = mode === "draw" && spec.acceptsTarget;
  const canGround = image !== null && !drawing && groundable(op, target) && !busy;
  const needsChoice = !drawing && ambiguous && candidates !== null && selected === REJECTED;
  const region = image === null ? PHRASE : regionFor();
  const canRun =
    image !== null &&
    ready({ op, imageSha256: image.sha256, target, content, colour }, region) &&
    !busy &&
    !needsChoice &&
    (!drawing || region.kind === "drawn");

  const chosenMask = region.kind === "phrase" ? null : region.mask;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8">
      <section className="flex flex-col gap-3">
        <ImageDrop
          onFile={(file) => void onFile(file)}
          busy={phase === "uploading"}
          replacing={image !== null}
        />
        {image !== null ? (
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            {image.width}×{image.height}
            {image.megapixels > 0 ? ` · ${image.megapixels.toFixed(1)} MP` : ""}
          </p>
        ) : null}
      </section>

      {imageUrl !== null && phase !== "done" ? (
        drawing && image !== null ? (
          <BrushCanvas
            imageUrl={imageUrl}
            aspect={image.width / image.height}
            history={strokes}
            onChange={setStrokes}
            base={tapMask}
            onTap={(point) => void onTap(point)}
            tapping={tapping}
          />
        ) : (
          <MaskPreview
            imageUrl={imageUrl}
            mask={chosenMask}
            alt={chosenMask === null ? "The picture you uploaded" : "The region that will change"}
          />
        )
      ) : null}

      {image !== null ? (
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2" role="group" aria-label="What to do">
            {OPERATIONS.map((choice) => (
              <button
                key={choice.op}
                type="button"
                aria-pressed={choice.op === op}
                onClick={() => {
                  setOp(choice.op);
                  // The region belonged to the previous operation, whether it was grounded
                  // for a phrase or painted by hand; keeping it would silently apply an
                  // approval the user never gave to this one.
                  forgetRegion();
                  setStrokes(EMPTY_HISTORY);
                  setTaps([]);
                  setTapMask(null);
                }}
                className={`rounded-md border px-3 py-2 text-sm ${
                  choice.op === op
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                    : "border-neutral-300 dark:border-neutral-700"
                }`}
              >
                {choice.label}
              </button>
            ))}
          </div>

          {spec.acceptsTarget ? (
            <div className="flex flex-wrap items-center gap-2" role="group" aria-label="How to choose the region">
              <button
                type="button"
                aria-pressed={mode === "describe"}
                onClick={() => setMode("describe")}
                className={`rounded-md border px-3 py-2 text-sm ${
                  mode === "describe"
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                    : "border-neutral-300 dark:border-neutral-700"
                }`}
              >
                Describe it
              </button>
              <button
                type="button"
                aria-pressed={mode === "draw"}
                onClick={() => setMode("draw")}
                className={`rounded-md border px-3 py-2 text-sm ${
                  mode === "draw"
                    ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                    : "border-neutral-300 dark:border-neutral-700"
                }`}
              >
                Draw it
              </button>
            </div>
          ) : null}

          {spec.examples.length > 0 && !drawing ? (
            <div className="flex flex-wrap gap-2" aria-label="Example prompts" role="group">
              {spec.examples.map(([exampleTarget, exampleContent]) => (
                <button
                  key={`${exampleTarget}|${exampleContent}`}
                  type="button"
                  onClick={() => {
                    setTarget(exampleTarget);
                    setContent(exampleContent);
                    forgetRegion();
                  }}
                  className="rounded-full border border-neutral-300 px-3 py-1 text-xs text-neutral-600 dark:border-neutral-700 dark:text-neutral-400"
                >
                  {exampleContent && exampleTarget
                    ? `${exampleTarget} → ${exampleContent}`
                    : exampleTarget || exampleContent}
                </button>
              ))}
            </div>
          ) : null}

          {spec.acceptsTarget && !drawing ? (
            <input
              value={target}
              onChange={(event) => {
                setTarget(event.target.value);
                forgetRegion();
              }}
              placeholder={spec.targetHint}
              aria-label="What to change"
              className="rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
            />
          ) : null}

          {spec.picksColour ? (
            // A colour input rather than a description, because this operation composites
            // a flat backdrop rather than generating one. Asking for prose here promised
            // something it cannot do, and until `EditSpec.colour` existed it silently
            // ignored the answer (TD-020).
            <label className="flex items-center gap-3 text-sm">
              <input
                type="color"
                value={colour}
                onChange={(event) => setColour(event.target.value)}
                aria-label="Backdrop colour"
                className="h-10 w-16 cursor-pointer rounded-md border border-neutral-300 bg-transparent dark:border-neutral-700"
              />
              <span className="text-neutral-600 dark:text-neutral-400">
                Backdrop colour <code className="text-xs">{colour}</code>
              </span>
            </label>
          ) : spec.needsContent ? (
            <input
              value={content}
              onChange={(event) => setContent(event.target.value)}
              placeholder={spec.contentHint}
              aria-label="What to put there"
              className="rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
            />
          ) : null}

          <div className="flex flex-wrap items-center gap-2">
            {spec.acceptsTarget && !drawing ? (
              <button
                type="button"
                onClick={() => void onGround()}
                disabled={!canGround}
                className="rounded-md border border-neutral-300 px-3 py-2 text-sm disabled:opacity-40 dark:border-neutral-700"
              >
                {phase === "grounding" ? "Looking…" : "Find it"}
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => void onRun()}
              disabled={!canRun}
              className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-white dark:text-neutral-900"
            >
              {phase === "running" ? "Working…" : "Run"}
            </button>
            {needsChoice ? (
              <span className="text-xs text-neutral-500 dark:text-neutral-400">
                Pick a region first, or describe it differently.
              </span>
            ) : null}
          </div>
        </section>
      ) : null}

      {candidates !== null && imageUrl !== null && image !== null ? (
        ambiguous || expanded ? (
          <CandidatePicker
            candidates={candidates}
            imageUrl={imageUrl}
            aspect={image.width / image.height}
            selected={selected}
            onSelect={(index) => {
              setSelected(index);
              // The measured reason the brush exists: the right region is absent from the
              // top five about one time in six, and a phrase the detector cannot ground
              // will not ground on a second attempt either.
              if (index === REJECTED) setMode("draw");
            }}
            phrase={target.trim()}
          />
        ) : (
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            {/* Not "found a clear match": the detector answers even for a phrase that
                matches nothing in the picture (TD-023), so the honest claim is about what
                will happen, not about how sure anything is. The preview above is the
                evidence, and this link is the way out. */}
            This is the region that will change.{" "}
            <button
              type="button"
              onClick={() => setExpanded(true)}
              className="underline underline-offset-2"
            >
              Not what you meant?
            </button>
          </p>
        )
      ) : null}

      {progress !== null && phase === "running" ? (
        <section aria-live="polite" className="flex flex-col gap-1">
          <div className="h-1.5 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
            <div
              className="h-full bg-blue-500 transition-[width]"
              style={{ width: `${Math.round(progress.progress * 100)}%` }}
            />
          </div>
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            {progress.detail || progress.state}
          </p>
          <StepTimeline steps={steps} />
        </section>
      ) : null}

      {error !== null ? (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      ) : null}

      {resultUrl !== null && phase === "done" ? (
        <section className="flex flex-col gap-3">
          {beforeUrl !== null ? (
            <BeforeAfter before={beforeUrl} after={resultUrl} />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={resultUrl}
              alt="The edited picture"
              className="max-h-[60vh] w-full rounded-lg object-contain"
            />
          )}
          <div className="flex flex-wrap gap-2">
            <a
              href={resultUrl}
              download="editgpt.png"
              className="rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700"
            >
              Download
            </a>
            <button
              type="button"
              onClick={() => onPickVersion(current)}
              className="rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700"
            >
              Keep editing this
            </button>
          </div>
          <StepTimeline steps={steps} />
        </section>
      ) : null}

      <VersionStrip
        versions={versions}
        current={current}
        urls={urls.current}
        onPick={onPickVersion}
      />
    </div>
  );
}
