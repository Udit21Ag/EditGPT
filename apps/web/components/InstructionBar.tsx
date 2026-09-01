"use client";

/**
 * Say what you want in a sentence, and see what was understood before anything runs.
 *
 * The interpretation is shown rather than applied silently, and it says **who** answered:
 * a rule, in a fraction of a millisecond and without leaving the machine, or the model.
 * That is the fast-path claim made visible instead of asserted — and when the answer is
 * wrong, correcting a chip costs a click rather than an edit.
 */

import type { PlanView } from "@/lib/api";

export interface InstructionBarProps {
  value: string;
  onChange: (value: string) => void;
  onInterpret: () => void;
  plan: PlanView | null;
  busy: boolean;
  disabled?: boolean;
}

const LANE: Record<PlanView["route"], string> = {
  rule: "matched a rule — no model was asked",
  model: "planned by the model",
  ask: "needs a clearer instruction",
};

export function InstructionBar({
  value,
  onChange,
  onInterpret,
  plan,
  busy,
  disabled = false,
}: InstructionBarProps) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex gap-2">
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            // Enter submits, because this is a one-line instruction and reaching for a
            // button to send a sentence is a keystroke nobody expects to need.
            if (event.key === "Enter" && !busy && value.trim().length > 0) onInterpret();
          }}
          placeholder="Say what you want — “remove the car on the left”"
          aria-label="What would you like to change?"
          disabled={disabled}
          className="flex-1 rounded-md border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
        />
        <button
          type="button"
          onClick={onInterpret}
          disabled={busy || disabled || value.trim().length === 0}
          className="rounded-md border border-neutral-300 px-3 py-2 text-sm disabled:opacity-50 dark:border-neutral-700"
        >
          {busy ? "Reading…" : "Interpret"}
        </button>
      </div>

      {plan !== null ? (
        <p
          role="status"
          className={`text-sm ${
            plan.route === "ask"
              ? "text-amber-700 dark:text-amber-400"
              : "text-neutral-600 dark:text-neutral-400"
          }`}
        >
          {plan.route === "ask" ? (
            plan.question
          ) : (
            <>
              <span className="font-medium">{plan.op}</span>
              {plan.target !== null ? <> → “{plan.target}”</> : null}
              {plan.content !== null ? <> → “{plan.content}”</> : null}
              <span className="ml-2 text-xs">({LANE[plan.route]})</span>
            </>
          )}
        </p>
      ) : null}
    </section>
  );
}
