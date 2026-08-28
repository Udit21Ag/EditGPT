"use client";

/**
 * Upload a picture, describe a change, confirm the region, watch it happen.
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
import { useAuth } from "@clerk/nextjs";
import {
  ApiError,
  createJob,
  getJob,
  groundPhrase,
  imageObjectUrl,
  streamJob,
  uploadImage,
  type Candidate,
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
import { BrushCanvas } from "./BrushCanvas";
import { CandidatePicker, REJECTED } from "./CandidatePicker";
import { MaskPreview } from "./MaskPreview";

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

export function EditWorkspace() {
  const { getToken } = useAuth();

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
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [ambiguous, setAmbiguous] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState(0);

  const [progress, setProgress] = useState<Progress | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  const stopStream = useRef<(() => void) | null>(null);
  const spec = specFor(op);

  // Blob URLs are held by the browser until revoked; without this every picture opened
  // in a session stays in memory for the life of the tab.
  useEffect(() => () => stopStream.current?.(), []);
  useEffect(() => () => { if (imageUrl !== null) URL.revokeObjectURL(imageUrl); }, [imageUrl]);
  useEffect(() => () => { if (resultUrl !== null) URL.revokeObjectURL(resultUrl); }, [resultUrl]);

  const reportError = (cause: unknown) => {
    setError(cause instanceof ApiError ? cause.message : "Something went wrong. Try again.");
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
      if (image === null || !hasMask(strokes)) return PHRASE;
      const drawn = maskFromStrokes(strokes.strokes, maskSize(image.width, image.height));
      return drawn === null ? PHRASE : { kind: "drawn", mask: drawn };
    }
    if (candidates !== null && selected >= 0) {
      const chosen = candidates[selected]?.mask;
      if (chosen !== undefined) return { kind: "chosen", mask: chosen };
    }
    return PHRASE;
  }, [mode, image, strokes, candidates, selected]);

  async function onFile(file: File) {
    forgetRegion();
    setError(null);
    setResultUrl(null);
    setPhase("uploading");
    try {
      const uploaded = await uploadImage(getToken, file);
      setImage(uploaded);
      setImageUrl(URL.createObjectURL(file));
      setPhase("idle");
    } catch (cause) {
      reportError(cause);
      setPhase("idle");
    }
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

  async function onRun() {
    if (image === null) return;
    setError(null);
    setResultUrl(null);
    setProgress(null);
    setPhase("running");

    const draft = { op, imageSha256: image.sha256, target, content, colour };

    try {
      const job = await createJob(getToken, buildJob(draft, regionFor()));
      stopStream.current = streamJob(getToken, job.id, (event) => {
        setProgress({ state: event.state, progress: event.progress, detail: event.detail });
        if (event.terminal) void finish(job.id);
      });
    } catch (cause) {
      reportError(cause);
      setPhase("idle");
    }
  }

  const finish = useCallback(
    async (id: string) => {
      stopStream.current?.();
      try {
        const done = await getJob(getToken, id);
        if (done.result_sha256 === null) {
          setError(done.error ?? "The edit finished without producing an image.");
          setPhase("idle");
          return;
        }
        setResultUrl(await imageObjectUrl(getToken, done.result_sha256));
        setPhase("done");
      } catch (cause) {
        reportError(cause);
        setPhase("idle");
      }
    },
    [getToken],
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
        <label className="flex w-fit cursor-pointer items-center gap-2 rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700">
          <input
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file !== undefined) void onFile(file);
            }}
          />
          {image === null ? "Choose a picture" : "Choose a different picture"}
        </label>
        {image !== null ? (
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            {image.width}×{image.height} · {image.megapixels.toFixed(1)} MP
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
        </section>
      ) : null}

      {error !== null ? (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      ) : null}

      {resultUrl !== null && phase === "done" ? (
        <section className="flex flex-col gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={resultUrl}
            alt="The edited picture"
            className="max-h-[60vh] w-full rounded-lg object-contain"
          />
          <a
            href={resultUrl}
            download="editgpt.png"
            className="w-fit rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700"
          >
            Download
          </a>
        </section>
      ) : null}
    </div>
  );
}
