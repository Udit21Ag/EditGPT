/**
 * Finding the picture in a drop or a paste.
 *
 * Both arrive as a `DataTransferItemList` carrying whatever the source felt like
 * attaching — a screenshot paste from a browser brings the image *and* an `text/html`
 * fragment describing it, and a drop from a file manager can bring several files at once.
 * Picking the first image and ignoring the rest is the whole job, and it is worth having
 * somewhere testable because the alternative is discovering it against a real clipboard.
 */

export const ACCEPTED = ["image/jpeg", "image/png", "image/webp", "image/avif"] as const;

function supported(type: string): boolean {
  return (ACCEPTED as readonly string[]).includes(type.toLowerCase());
}

/**
 * The first supported image, or null.
 *
 * Null is a normal outcome — a text paste, a dragged link — and the caller says nothing
 * rather than showing an error for something the user did not intend as an upload.
 */
export function pickImage(items: DataTransferItemList | null | undefined): File | null {
  if (items === null || items === undefined) return null;
  for (const item of Array.from(items)) {
    if (item.kind !== "file") continue;
    const file = item.getAsFile();
    if (file !== null && supported(file.type)) return file;
  }
  return null;
}

/** The same, for a drop that carried `files` rather than `items`. */
export function pickFile(files: FileList | null | undefined): File | null {
  if (files === null || files === undefined) return null;
  for (const file of Array.from(files)) if (supported(file.type)) return file;
  return null;
}
