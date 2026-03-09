// frontend/src/App.tsx
import { useEffect, useMemo, useRef, useState } from "react";

type Project = { id: string; name: string };

type WsMsg =
  | { type: "partial"; project_id: string; text: string; seq: number }
  | { type: "project_transcript"; project_id: string; text: string }
  | { type: "error"; message: string };

const DEFAULT_PROJECTS: Project[] = [
  { id: "proj-1", name: "Quick check AI speech recognition" },
  { id: "proj-2", name: "Project 2" },
  { id: "proj-3", name: "Project 3" },
];

function makeWsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  // backend on port 8000
  return `${proto}://${location.hostname}:8000/ws/transcribe`;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("FileReader failed"));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      // data:audio/webm;codecs=opus;base64,XXXX
      const comma = dataUrl.indexOf(",");
      resolve(comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl);
    };
    reader.readAsDataURL(blob);
  });
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(DEFAULT_PROJECTS);
  const [activeProjectId, setActiveProjectId] = useState(projects[0]?.id ?? "proj-1");
  const [language, setLanguage] = useState<"auto" | "fi" | "en">("auto");

  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);

  const [linesByProject, setLinesByProject] = useState<Record<string, Array<{ seq: number; text: string; ts: number }>>>({});

  // --- REFS ---
  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const seqRef = useRef(0);
  
  // NEW REFS ADDED HERE
  const shouldRecordRef = useRef(false);
  const chunkMsRef = useRef(2500); // 2.5s is a good starting point
  const activeProjectIdRef = useRef(activeProjectId);
const languageRef = useRef(language);

useEffect(() => { activeProjectIdRef.current = activeProjectId; }, [activeProjectId]);
useEffect(() => { languageRef.current = language; }, [language]);
  const sessionIdRef = useRef<string>(`session-${Date.now()}`);
const sessionId = sessionIdRef.current;

  const activeProject = useMemo(() => projects.find(p => p.id === activeProjectId), [projects, activeProjectId]);

  // Connect WS
  useEffect(() => {
    const ws = new WebSocket(makeWsUrl());
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (ev) => {
      try {
        const msg: WsMsg = JSON.parse(ev.data);
        if (msg.type === "partial") {
          setLinesByProject(prev => {
            const arr = prev[msg.project_id] ? [...prev[msg.project_id]] : [];
            arr.push({ seq: msg.seq, text: msg.text, ts: Date.now() });
            return { ...prev, [msg.project_id]: arr };
          });
        }
      } catch {
        // ignore
      }
    };

    return () => ws.close();
  }, []);
  
  // NEW HELPER ADDED HERE
  function pickMimeType(): string | undefined {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
    ];
    for (const c of candidates) {
      // @ts-ignore
      if (window.MediaRecorder?.isTypeSupported?.(c)) return c;
    }
    return undefined;
  }

  // NEW START RECORDING LOGIC
  async function startRecording() {
    if (recording) return;
    if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert("WebSocket not connected to backend.");
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    shouldRecordRef.current = true;
    setRecording(true);

    const mimeType = pickMimeType();

    const startOneChunk = () => {
      if (!shouldRecordRef.current || !streamRef.current) return;

      const recorder = new MediaRecorder(
        streamRef.current,
        mimeType ? { mimeType } : undefined
      );
      recorderRef.current = recorder;

      recorder.ondataavailable = async (e) => {
        if (!e.data || e.data.size === 0) return;
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const seq = seqRef.current++;
        const data_b64 = await blobToBase64(e.data);

        ws.send(
          JSON.stringify({
            type: "audio_chunk",
            session_id: sessionId,
            project_id: activeProjectIdRef.current,
language: languageRef.current,
            seq,
            mime: e.data.type || mimeType || "audio/webm",
            data_b64,
          })
        );
      };

      recorder.onstop = () => {
        // Immediately start the next chunk if still recording
        if (shouldRecordRef.current) startOneChunk();
      };

      recorder.start();

      // stop after chunkMs to finalize container so ffmpeg can decode
      setTimeout(() => {
        try {
          if (recorder.state === "recording") recorder.stop();
        } catch {}
      }, chunkMsRef.current);
    };

    startOneChunk();
  }

  // NEW STOP RECORDING LOGIC
  function stopRecording() {
    if (!recording) return;

    shouldRecordRef.current = false;

    try {
      if (recorderRef.current && recorderRef.current.state === "recording") {
        recorderRef.current.stop();
      }
    } catch {}
    recorderRef.current = null;

    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;

    setRecording(false);
  }

  const activeLines = linesByProject[activeProjectId] ?? [];

  return (
    <div className="h-full bg-slate-950 text-slate-100">
      <div className="h-full grid grid-cols-[320px_1fr]">
        <aside className="border-r border-slate-800 p-4 flex flex-col gap-4">
          <div>
            <div className="text-lg font-semibold">QuickCheck AI</div>
            <div className="text-xs text-slate-400">
              {connected ? "● Connected to backend" : "○ Disconnected"} • {recording ? "🎙️ Recording" : "Idle"}
            </div>
            <div className="text-[11px] text-slate-500 mt-1">
              Session: {sessionId}
            </div>
          </div>

          <div className="bg-slate-900/60 rounded-2xl p-3 border border-slate-800">
            <div className="text-xs text-slate-400 mb-2">Language</div>
            <select
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2"
              value={language}
              onChange={(e) => setLanguage(e.target.value as any)}
            >
              <option value="auto">Auto</option>
              <option value="fi">Finnish (fi)</option>
              <option value="en">English (en)</option>
            </select>
          </div>

          <div className="bg-slate-900/60 rounded-2xl p-3 border border-slate-800">
            <div className="text-xs text-slate-400 mb-2">Controls</div>
            {!recording ? (
              <button
                onClick={startRecording}
                className="w-full rounded-xl bg-slate-200 text-slate-950 font-semibold py-2 hover:bg-white"
              >
                Start mic
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="w-full rounded-xl bg-red-400 text-slate-950 font-semibold py-2 hover:bg-red-300"
              >
                Stop mic
              </button>
            )}
            <div className="text-[11px] text-slate-500 mt-2">
              Switch projects anytime — new audio chunks will be tagged to the active project.
            </div>
          </div>

          <div className="flex-1">
            <div className="text-xs text-slate-400 mb-2">Projects</div>
            <div className="flex flex-col gap-2">
              {projects.map((p) => {
                const active = p.id === activeProjectId;
                return (
                  <button
                    key={p.id}
                    onClick={() => setActiveProjectId(p.id)}
                    className={[
                      "text-left rounded-2xl border p-3 transition",
                      active
                        ? "border-slate-200 bg-slate-200 text-slate-950"
                        : "border-slate-800 bg-slate-900/40 hover:bg-slate-900"
                    ].join(" ")}
                  >
                    <div className="font-semibold">{p.name}</div>
                    <div className={active ? "text-xs text-slate-700" : "text-xs text-slate-400"}>
                      {p.id} • lines: {(linesByProject[p.id]?.length ?? 0)}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400">Active project</div>
              <div className="text-2xl font-bold">{activeProject?.name ?? activeProjectId}</div>
            </div>
            <div className="text-xs text-slate-400">
              Captions update every ~2.5s
            </div>
          </div>

          <div className="flex-1 rounded-3xl border border-slate-800 bg-slate-900/30 overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between">
              <div className="font-semibold">Live transcript</div>
              <div className="text-xs text-slate-400">Mic: {recording ? "ON" : "OFF"}</div>
            </div>

            <div className="p-5 h-full overflow-auto space-y-3">
              {activeLines.length === 0 ? (
                <div className="text-slate-400">
                  Start the mic and speak. The transcript will appear here.
                </div>
              ) : (
                activeLines.slice(-200).map((l, i) => (
                  <div key={i} className="rounded-2xl border border-slate-800 bg-slate-950/50 p-3">
                    <div className="text-xs text-slate-400 mb-1">
                      #{l.seq} • {new Date(l.ts).toLocaleTimeString()}
                    </div>
                    <div className="text-slate-100">{l.text || <span className="text-slate-500">(no text)</span>}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}