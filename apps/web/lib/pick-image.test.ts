/**
 * Choosing the picture out of a drop or a paste.
 *
 * The fixtures mirror what real sources send, which is the point: a screenshot pasted
 * from a browser carries an `text/html` fragment alongside the image, and a drop from a
 * file manager can carry several files.
 */

import { describe, expect, it } from "vitest";
import { pickFile, pickImage } from "./pick-image";

function file(name: string, type: string): File {
  return new File(["x"], name, { type });
}

function items(...entries: (File | { kind: string; type: string })[]): DataTransferItemList {
  const list = entries.map((entry) =>
    entry instanceof File
      ? { kind: "file", type: entry.type, getAsFile: () => entry }
      : { ...entry, getAsFile: () => null },
  );
  return list as unknown as DataTransferItemList;
}

describe("pasting", () => {
  it("takes the image and ignores the markup pasted with it", () => {
    const png = file("shot.png", "image/png");
    expect(pickImage(items({ kind: "string", type: "text/html" }, png))).toBe(png);
  });

  it("takes the first image when several arrive", () => {
    const first = file("a.png", "image/png");
    expect(pickImage(items(first, file("b.jpg", "image/jpeg")))).toBe(first);
  });

  it("accepts every format the gateway does", () => {
    for (const type of ["image/jpeg", "image/png", "image/webp", "image/avif"]) {
      expect(pickImage(items(file("x", type)))).not.toBeNull();
    }
  });

  it("is not fooled by the case of the type", () => {
    expect(pickImage(items(file("x.PNG", "IMAGE/PNG")))).not.toBeNull();
  });

  it("returns null for a text paste, which is not an error", () => {
    expect(pickImage(items({ kind: "string", type: "text/plain" }))).toBeNull();
  });

  it("refuses a format the gateway would reject anyway", () => {
    // Better here than as a 415 after uploading several megabytes.
    expect(pickImage(items(file("clip.gif", "image/gif")))).toBeNull();
    expect(pickImage(items(file("doc.pdf", "application/pdf")))).toBeNull();
  });

  it("survives a paste that carried nothing", () => {
    expect(pickImage(null)).toBeNull();
    expect(pickImage(undefined)).toBeNull();
  });
});

describe("dropping", () => {
  it("takes the first supported file and skips the others", () => {
    const list = [file("notes.txt", "text/plain"), file("photo.jpg", "image/jpeg")];
    expect(pickFile(list as unknown as FileList)?.name).toBe("photo.jpg");
  });

  it("returns null when a drop carried no picture", () => {
    expect(pickFile([file("a.txt", "text/plain")] as unknown as FileList)).toBeNull();
    expect(pickFile(null)).toBeNull();
  });
});
