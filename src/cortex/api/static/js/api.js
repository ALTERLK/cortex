/* Everything that talks to the server lives here. */

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function check(response) {
  if (response.ok) return response;
  const body = await response.json().catch(() => ({}));
  throw new ApiError(body.detail || `Server error (${response.status})`, response.status);
}

export const getJSON = async (path) => (await check(await fetch(path))).json();

/**
 * Parse a Server-Sent-Events stream from fetch(), yielding {event, data}.
 *
 * The browser's EventSource only supports GET, and /ask/stream needs a POST
 * with a JSON body, so the response body is read by hand. The framing is
 * simple: frames end with a blank line and carry an `event:` and a `data:`
 * line. Bytes are buffered until a whole frame is available, because a
 * network chunk can end in the middle of one.
 */
export async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);

      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

/** Ask a question; `onEvent(event, data)` is called for every stream event. */
export async function ask(payload, { signal, onEvent }) {
  const response = await check(await fetch("/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  }));
  for await (const { event, data } of readSSE(response)) onEvent(event, data);
}

/** Upload one file. The raw file is the request body; returns {job_id, source}. */
export async function uploadDocument(file) {
  const response = await check(await fetch(`/documents/${encodeURIComponent(file.name)}`, {
    method: "PUT",
    body: file,
  }));
  return response.json();
}

/** Poll an ingest job until it finishes; `onProgress(done, total)` on each poll. */
export async function waitForJob(jobId, onProgress) {
  for (;;) {
    const job = await getJSON(`/ingest/${jobId}`);
    onProgress?.(job.files_done, job.files_total);
    if (job.status === "done") return job;
    if (job.status === "failed") throw new ApiError(job.error || "Indexing failed", 500);
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
}

export async function deleteDocument(source) {
  // Each path segment is encoded on its own so the "/" separators survive.
  const path = source.split("/").map(encodeURIComponent).join("/");
  await check(await fetch(`/documents/${path}`, { method: "DELETE" }));
}
