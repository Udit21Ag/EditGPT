/**
 * The instruction bar, in a real browser.
 *
 * What matters here is not that a form submits: it is that the user is told **who**
 * answered. The fast-path claim — most instructions never reach a model — is only
 * checkable by someone using the thing if the page says so.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { PlanView } from "@/lib/api";
import { InstructionBar } from "./InstructionBar";

function view(over: Partial<PlanView> = {}): PlanView {
  return {
    route: "rule",
    op: "remove",
    target: "car on the left",
    content: null,
    colour: null,
    question: null,
    seconds: 0.0001,
    tokens: 0,
    ...over,
  };
}

describe("InstructionBar", () => {
  it("says that a rule answered, and that nothing was asked", () => {
    render(
      <InstructionBar
        value="remove the car on the left"
        onChange={vi.fn()}
        onInterpret={vi.fn()}
        plan={view()}
        busy={false}
      />,
    );

    const status = screen.getByRole("status");
    expect(status.textContent).toContain("remove");
    expect(status.textContent).toContain("car on the left");
    expect(status.textContent).toContain("no model was asked");
  });

  it("distinguishes the model's answer from a rule's", () => {
    render(
      <InstructionBar
        value="that lamppost ruins it"
        onChange={vi.fn()}
        onInterpret={vi.fn()}
        plan={view({ route: "model", target: "lamppost", tokens: 201 })}
        busy={false}
      />,
    );

    expect(screen.getByRole("status").textContent).toContain("planned by the model");
  });

  it("shows the question when the instruction could not be read", () => {
    render(
      <InstructionBar
        value="make it nicer"
        onChange={vi.fn()}
        onInterpret={vi.fn()}
        plan={view({ route: "ask", op: null, target: null, question: "Try naming the operation." })}
        busy={false}
      />,
    );

    expect(screen.getByRole("status").textContent).toContain("Try naming the operation");
  });

  it("submits on Enter, because a sentence should not need a button", () => {
    const onInterpret = vi.fn();
    render(
      <InstructionBar
        value="remove the car"
        onChange={vi.fn()}
        onInterpret={onInterpret}
        plan={null}
        busy={false}
      />,
    );

    fireEvent.keyDown(screen.getByLabelText("What would you like to change?"), { key: "Enter" });
    expect(onInterpret).toHaveBeenCalledOnce();
  });

  it("does not submit an empty instruction, or one already in flight", () => {
    const onInterpret = vi.fn();
    const { rerender } = render(
      <InstructionBar
        value="   "
        onChange={vi.fn()}
        onInterpret={onInterpret}
        plan={null}
        busy={false}
      />,
    );
    fireEvent.keyDown(screen.getByLabelText("What would you like to change?"), { key: "Enter" });

    rerender(
      <InstructionBar
        value="remove the car"
        onChange={vi.fn()}
        onInterpret={onInterpret}
        plan={null}
        busy={true}
      />,
    );
    fireEvent.keyDown(screen.getByLabelText("What would you like to change?"), { key: "Enter" });

    expect(onInterpret).not.toHaveBeenCalled();
  });
});
