"use client";

/**
 * What the job did, step by step, as it happens.
 *
 * The worker already records every state change with a timestamp and a detail string —
 * including the passes it tried and rolled back, which `harness/architecture.md` calls a
 * finding rather than noise. None of it reached a user; the progress bar showed a number
 * and threw the reasoning away.
 *
 * Live steps arrive over SSE and terminal ones come from the job itself, so this takes a
 * plain list and does not care which. `aria-live` on the container, not on each row: a
 * screen reader should hear the newest step, not re-read the whole list every second.
 */

export interface Step {
  readonly state: string;
  readonly detail: string;
  readonly at?: string;
}

function clock(at: string | undefined): string {
  if (at === undefined) return "";
  const when = new Date(at);
  return Number.isNaN(when.getTime()) ? "" : when.toLocaleTimeString();
}

export function StepTimeline({ steps }: { steps: readonly Step[] }) {
  if (steps.length === 0) return null;
  return (
    <ol aria-label="What the job did" aria-live="polite" className="flex flex-col gap-1 text-xs">
      {steps.map((step, index) => (
        <li key={`${step.state}-${index}`} className="flex items-baseline gap-2">
          <span
            aria-hidden
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              index === steps.length - 1 ? "bg-blue-500" : "bg-neutral-400"
            }`}
          />
          <span className="font-medium">{step.state}</span>
          {step.detail ? (
            <span className="text-neutral-600 dark:text-neutral-400">{step.detail}</span>
          ) : null}
          <span className="ml-auto tabular-nums text-neutral-400">{clock(step.at)}</span>
        </li>
      ))}
    </ol>
  );
}
