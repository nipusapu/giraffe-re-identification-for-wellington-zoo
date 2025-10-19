"use client";

import Description from "./_components/description";
import Nav from "./_components/nav";
import Button from "./_components/button";
import React, { useEffect, useRef, useState } from "react";
import { AiOutlineCloudUpload } from "react-icons/ai";
import Image from "next/image";

type UploadOk = { reid_id: number; image_id: string; status: string };
type UploadErr = { detail?: string; error?: string };

type ResultPayload = {
  id: number;
  status: string; // queued | detecting | no_detection | reidentifying | completed | error
  predicted_animal?: string | null;
  display_name?: string | null;
  description?: string | null;
  bio?: string | null;
  about?: string | null;
  votes?: Record<string, number>;
};

type Timing = {
  jobId: number | null;
  t0: number;
  uploadEnd?: number;
  pollStart?: number;
  done?: number;
  totalMs?: number;
  wallStart?: number;
  wallDone?: number;
};

const descriptionArr = [
  { title: "KIA ORA", description: "World First Giraffe Re-Identification System For Zoo Visitors" },
  { title: "What is Re-Identification?", description: "It is a process which focuses on ID Objects any time. This is same as your smartphone ID your face." },
  { title: "What is this app?", description: "This app mainly focuses on ID Giraffes in zoo. We want you to feel like an explorer than a visitor." },
  { title: "How it Works?", description: "You just need to upload a clear picture of the Giraffe’s body. Then app will give you bio of the Giraffe." },
  { title: "Ready to Explore?", description: "" },
];

const TERMINAL = new Set(["completed", "error", "no_detection"] as const);
const PRETTY: Record<string, string> = {
  queued: "Queued",
  detecting: "Detecting",
  no_detection: "No detection",
  reidentifying: "Re-Identifying",
  completed: "Completed",
  error: "Error",
};

// Safari-safe time helpers
const perfNow = () =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

export default function Home() {
  const [count, setCount] = useState<number>(0);
  const [phase, setPhase] = useState<"idle" | "uploading" | "polling" | "done">("idle");

  const [jobId, setJobId] = useState<number | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<ResultPayload | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [timing, setTiming] = useState<Timing | null>(null);
  const lastStatusRef = useRef<string | null>(null);

  const nextSlide = () => {
    if (count < 5) setCount((c) => c + 1);
  };

  // ===== Select (no upload yet) =====
  const onFileChosen: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);

    setSelectedFile(file);
    setPreviewUrl(url);

    setError(null);
    setFinalResult(null);
    setStatus(null);
    setPhase("idle");
  };

  // ===== Upload the selected file =====
  const uploadSelected = async () => {
    if (!selectedFile) return;
    setPhase("uploading");

    const t0 = perfNow();
    const wallStart = Date.now();
    setTiming({ jobId: null, t0, wallStart });

    console.log("%c[ReID] Upload start", "color:#2563eb;font-weight:700", { wallStart: new Date(wallStart).toISOString() });

    try {
      const fd = new FormData();
      fd.append("image", selectedFile);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: fd,
        // explicit headers help Safari networking stack
        headers: { "Accept": "application/json", "Cache-Control": "no-cache" as any },
        cache: "no-store",
      });

      // In case a proxy returns non-JSON on error in Safari
      const json = (await res.json().catch(() => ({}))) as UploadOk & UploadErr;

      if (!res.ok) {
        console.log("%c[ReID] Upload failed", "color:#dc2626;font-weight:700", json);
        setError(json?.detail || json?.error || `Upload failed (${res.status})`);
        setPhase("idle");
        return;
      }

      const uploadEnd = perfNow();
      setTiming((t) => t ? { ...t, jobId: json.reid_id ?? null, uploadEnd } : null);

      setJobId(json.reid_id);
      setImageId(json.image_id);
      setStatus(json.status ?? "queued");

      console.log("%c[ReID] Upload done → job created", "color:#16a34a;font-weight:700", {
        jobId: json.reid_id, imageId: json.image_id, status: json.status ?? "queued", uploadMs: +(uploadEnd - t0).toFixed(1)
      });

      const pollStart = perfNow();
      setTiming((t) => (t ? { ...t, pollStart } : null));
      console.log("%c[ReID] Polling start", "color:#7c3aed;font-weight:700", { jobId: json.reid_id });

      setPhase("polling");
      if (count < 4) setCount(4);
    } catch (e) {
      console.log("%c[ReID] Network error during upload", "color:#dc2626;font-weight:700", e);
      setError("Network error");
      setPhase("idle");
    }
  };

  // ===== Clear selection =====
  const clearSelection = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setSelectedFile(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  // ===== Polling =====
  useEffect(() => {
    if (phase !== "polling" || !jobId) return;

    const tick = async () => {
      try {
        const res = await fetch(`/api/result/${jobId}?_=${Date.now()}`, {
          cache: "no-store",
          headers: { "Cache-Control": "no-cache" as any },
        });
        if (!res.ok) return;
        const data = (await res.json()) as ResultPayload;

        if (data.status !== lastStatusRef.current) {
          console.log("%c[ReID] Status", "color:#0ea5e9;font-weight:700", {
            jobId, status: data.status, pretty: PRETTY[data.status] ?? data.status
          });
          lastStatusRef.current = data.status;
        }

        setStatus(data.status);

        if (TERMINAL.has(data.status as any)) {
          setFinalResult(data);
          setPhase("done");

          const done = perfNow();
          setTiming((t) => {
            if (!t) return t;
            const totalMs = done - t.t0;
            const wallDone = Date.now();
            console.log("%c[ReID] DONE", "color:#16a34a;font-weight:700", {
              jobId, status: data.status, totalMs: +totalMs.toFixed(1),
              startedAt: new Date(t.wallStart!).toISOString(),
              finishedAt: new Date(wallDone).toISOString(),
            });
            return { ...t, done, wallDone, totalMs };
          });
        }
      } catch (e) {
        console.log("%c[ReID] Polling error (ignored)", "color:#f59e0b;font-weight:700", e);
      }
    };

    tick();
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, [phase, jobId]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const reset = (goToReady?: boolean) => {
    if (goToReady) setCount(4);
    else if (count < 4) setCount(4);

    setPhase("idle");
    setJobId(null);
    setImageId(null);
    setStatus(null);
    setFinalResult(null);
    setError(null);
    setShowDetails(false);
    lastStatusRef.current = null;
    clearSelection();
    setTiming(null);
    console.log("%c[ReID] Reset", "color:#6b7280;font-weight:700");
  };

  const statusLabel =
    phase === "polling" ? `${PRETTY[status ?? "queued"] ?? status ?? "…"}`
      : "Uploading …";

  const prettyName =
    finalResult?.display_name?.trim() ||
    finalResult?.predicted_animal?.trim() ||
    "this giraffe";

  const userDescription =
    (finalResult?.description && finalResult.description.trim()) ||
    (finalResult?.bio && finalResult.bio.trim()) ||
    (finalResult?.about && finalResult.about.trim()) ||
    `Meet ${prettyName}! Thanks for helping us learn more about our giraffes.`;

  return (
    <div className="h-screen flex flex-col">
      <Nav />

      {(phase === "uploading" || phase === "polling") && (
        <div className="flex flex-1 justify-center items-center text-center">
          <div className="flex flex-col items-center">
            <img
              src="/loadingScreen.gif"
              alt="Loading..."
              className="w-[66vw] max-w-[1600px] h-auto max-h-[66vh] object-contain mb-4"
            />
            <div className="flex justify-center items-center bg-white w-80 p-4 text-black rounded shadow">
              <div className="text-base">
                {statusLabel}
                {phase === "polling" && jobId != null && (
                  <span className="block text-xs text-gray-600 mt-1">
                    ReID Job ID: {jobId}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {(phase === "idle" || phase === "done") && (
        <>
          {count < 5 ? (
            <div className="flex flex-1 justify-center items-center text-center">
              <div className="bg-amber-700/60 xl:p-4 lg:p-4 md:p-4 sm:p-2 xs:p-2 h-3/4 w-5/6 bg-opacity-90 relative">
                <Description
                  title={descriptionArr[count].title}
                  description={descriptionArr[count].description}
                />

                {count === 4 && (
                  <div className="mx-auto w-full max-w-[420px] sm:max-w-[560px] md:max-w-[900px] px-4">
                    <div className="grid grid-cols-2 gap-2 sm:gap-3 md:gap-4">
                      <div className="relative overflow-hidden rounded-xl bg-black/10 aspect-[9/16] md:aspect-[9/7]">
                        <Image
                          src="/images/sample1.jpg"
                          alt="Sample photo 1"
                          fill
                          priority
                          className="object-contain"
                          sizes="(max-width: 640px) 46vw, 45vw"
                        />
                      </div>
                      <div className="relative overflow-hidden rounded-xl bg-black/10 aspect-[9/16] md:aspect-[9/7]">
                        <Image
                          src="/images/sample2.jpg"
                          alt="Sample photo 2"
                          fill
                          className="object-contain"
                          sizes="(max-width: 640px) 46vw, 45vw"
                        />
                      </div>
                    </div>

                    <p className="text-center text-[14px] text-white/80 mt-2">
                      Example photos — Single, clear body, good lighting.
                    </p>
                  </div>
                )}

                <Button
                  className="absolute left-1/2 -translate-x-1/2 bottom-5 w-40"
                  title={count === 4 ? "Try It" : "Next"}
                  onClick={nextSlide}
                />
              </div>
            </div>
          ) : (
            <div className="flex flex-1 justify-center items-center text-center">
              {phase === "done" && finalResult ? (
                <div className="bg-white/95 text-black p-6 rounded-xl shadow w-[38rem] text-left">
                  <h2 className="text-2xl font-bold mb-4">You have captured {prettyName}!</h2>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="rounded overflow-hidden border bg-gray-50 flex items-center justify-center">
                      {previewUrl ? (
                        <img src={previewUrl} alt="Your uploaded photo" className="max-h-64 object-contain" />
                      ) : (
                        <div className="p-6 text-sm text-gray-500 text-center">(photo preview unavailable)</div>
                      )}
                    </div>

                    <div>
                      <p className="text-sm leading-relaxed whitespace-pre-line">{userDescription}</p>

                      <div className="mt-4">
                        <button
                          onClick={() => setShowDetails((v) => !v)}
                          className="text-xs px-3 py-1.5 rounded bg-gray-200 hover:bg-gray-300 transition"
                        >
                          {showDetails ? "Hide details" : "Show details"}
                        </button>
                        {showDetails && (
                          <div className="mt-3 text-xs text-gray-700 space-y-1">
                            <div><span className="font-medium">Job ID:</span> {finalResult.id}</div>
                            <div><span className="font-medium">Status:</span> {PRETTY[finalResult.status] ?? finalResult.status}</div>
                            {imageId && (<div><span className="font-medium">Image ID:</span> {imageId}</div>)}
                            {timing?.totalMs != null && (
                              <div><span className="font-medium">Total time (client):</span> {(timing.totalMs / 1000).toFixed(1)}s</div>
                            )}
                            {timing?.wallStart && (
                              <div><span className="font-medium">Started:</span> {new Date(timing.wallStart).toLocaleTimeString()}</div>
                            )}
                            {timing?.wallDone && (
                              <div><span className="font-medium">Completed:</span> {new Date(timing.wallDone).toLocaleTimeString()}</div>
                            )}
                            {finalResult.votes && Object.keys(finalResult.votes).length > 0 && (
                              <div>
                                <div className="font-medium">Votes</div>
                                <ul className="list-disc ml-5">
                                  {Object.entries(finalResult.votes).map(([k, v]) => (
                                    <li key={k}><span className="font-mono">{k}</span>: {v}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 flex gap-3 justify-end">
                    <button
                      onClick={() => reset(finalResult?.status === "no_detection")}
                      className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition"
                    >
                      Upload Another
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  {previewUrl ? (
                    <>
                      <img
                        src={previewUrl}
                        alt="Selected image preview"
                        className="w-[66vw] max-w-[1600px] h-auto max-h-[66vh] object-contain mb-4 rounded-xl bg-black/10"
                      />

                      <div className="flex gap-2">
                        <button
                          onClick={uploadSelected}
                          className="px-4 py-2 rounded-lg bg-green-600 text-white hover:bg-green-700 transition"
                        >
                          Upload
                        </button>

                        {/* Safari-safe: label triggers hidden input without programmatic click */}
                        <label
                          htmlFor="file-upload"
                          className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition cursor-pointer"
                        >
                          Choose Another
                        </label>

                        <button
                          onClick={clearSelection}
                          className="px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700 transition"
                        >
                          Remove
                        </button>
                      </div>

                      <input
                        id="file-upload"
                        ref={fileRef}
                        type="file"
                        accept="image/*"
                        // visually hidden but present for Safari; avoid display:none
                        className="absolute w-px h-px -m-px overflow-hidden clip-rect"
                        onChange={onFileChosen}
                      />
                    </>
                  ) : (
                    <>
                      <img
                        src="/uploading.gif"
                        alt="Uploading animation"
                        className="w-[66vw] max-w-[1600px] h-auto max-h-[66vh] object-contain mb-4"
                      />

                      {/* Whole card is a label that opens the picker in Safari */}
                      <label
                        htmlFor="file-upload"
                        className="flex items-center bg-white w-64 p-4 text-black rounded-lg shadow cursor-pointer select-none"
                        aria-label="Upload an image"
                        title="Choose an image"
                      >
                        <span className="mr-4 underline">Upload Here</span>
                        <AiOutlineCloudUpload size={30} className="ml-auto" />
                      </label>

                      <input
                        id="file-upload"
                        ref={fileRef}
                        type="file"
                        accept="image/*"
                        className="absolute w-px h-px -m-px overflow-hidden clip-rect"
                        onChange={onFileChosen}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-red-600 text-white px-4 py-2 rounded shadow">
              {error}
              <button className="ml-3 underline" onClick={() => setError(null)}>
                dismiss
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
