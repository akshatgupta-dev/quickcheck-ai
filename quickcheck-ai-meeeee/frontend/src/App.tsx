import "./App.css";
import { useEffect, useMemo, useRef, useState } from "react";

type Project = { id: string; name: string };
type TranscriptLine = { seq: number; text: string; ts: number; kind: "partial" | "final" };

type WsMsg =
  | { type: "ready"; sample_rate: number; frame_ms: number }
  | { type: "partial"; project_id: string; text: string; seq: number; is_final: false }
  | { type: "final"; project_id: string; text: string; seq: number; is_final: true }
  | { type: "project_transcript"; project_id: string; text: string }
  | { type: "error"; message: string };

const DEFAULT_PROJECTS: Project[] = [
  { id: "proj-1", name: "Quick check AI speech recognition" },
  { id: "proj-2", name: "Project 2" },
  { id: "proj-3", name: "Project 3" },
];

function makeWsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.hostname}:8000/ws/transcribe`;
}

function downsampleFloat32(input: Float32Array, inputSampleRate: number, targetSampleRate: number): Float32Array {
  if (targetSampleRate === inputSampleRate) return input;
  if (targetSampleRate > inputSampleRate) return input;

  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.max(1, Math.round(input.length / ratio));
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetInput = 0;

  while (offsetResult < result.length) {
    const nextOffsetInput = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;

    for (let i = offsetInput; i < Math.min(nextOffsetInput, input.length); i += 1) {
      accum += input[i];
      count += 1;
    }

    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetInput = nextOffsetInput;
  }

  return result;
}

function float32ToInt16(float32: Float32Array): Int16Array {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return int16;
}

function makeBinaryAudioMessage(audioBuffer: Int16Array, metadata: Record<string, unknown>): ArrayBuffer {
  const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
  const output = new Uint8Array(4 + metadataBytes.byteLength + audioBuffer.byteLength);
  new DataView(output.buffer).setUint32(0, metadataBytes.byteLength, true);
  output.set(metadataBytes, 4);
  output.set(new Uint8Array(audioBuffer.buffer), 4 + metadataBytes.byteLength);
  return output.buffer;
}

export default function App() {
  const [projects] = useState<Project[]>(DEFAULT_PROJECTS);
  const [activeProjectId, setActiveProjectId] = useState(projects[0]?.id ?? "proj-1");
  const [language, setLanguage] = useState<"auto" | "fi" | "en">("auto");
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [statusText, setStatusText] = useState("Idle");
  const [linesByProject, setLinesByProject] = useState<Record<string, TranscriptLine[]>>({});

  const wsRef = useRef<WebSocket | null>(null);
  const seqRef = useRef(0);
  const sessionIdRef = useRef<string>(`session-${Date.now()}`);

  const activeProjectIdRef = useRef(activeProjectId);
  const languageRef = useRef(language);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
  }, [activeProjectId]);

  useEffect(() => {
    languageRef.current = language;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "set_context",
          session_id: sessionIdRef.current,
          project_id: activeProjectIdRef.current,
          language: languageRef.current,
          seq: seqRef.current,
        })
      );
    }
  }, [activeProjectId, language]);

  const activeProject = useMemo(() => projects.find((p) => p.id === activeProjectId), [projects, activeProjectId]);
  const activeLines = linesByProject[activeProjectId] ?? [];

  useEffect(() => {
    const ws = new WebSocket(makeWsUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setStatusText("Backend connected");
      ws.send(
        JSON.stringify({
          type: "config",
          session_id: sessionIdRef.current,
          project_id: activeProjectIdRef.current,
          language: languageRef.current,
          sample_rate: 16000,
          seq: seqRef.current,
        })
      );
    };

    ws.onclose = () => {
      setConnected(false);
      setReady(false);
      setStatusText("Backend disconnected");
    };

    ws.onmessage = (event) => {
      const msg: WsMsg = JSON.parse(event.data);

      if (msg.type === "ready") {
        setReady(true);
        setStatusText(`Streaming ready • ${msg.sample_rate} Hz / ${msg.frame_ms} ms`);
        return;
      }

      if (msg.type === "partial") {
        setLinesByProject((prev) => {
          const current = prev[msg.project_id] ? [...prev[msg.project_id]] : [];
          const filtered = current.filter((line) => !(line.kind === "partial" && line.seq === msg.seq));
          filtered.push({ seq: msg.seq, text: msg.text, ts: Date.now(), kind: "partial" });
          return { ...prev, [msg.project_id]: filtered.slice(-300) };
        });
        return;
      }

      if (msg.type === "final") {
        setLinesByProject((prev) => {
          const current = prev[msg.project_id] ? [...prev[msg.project_id]] : [];
          const filtered = current.filter((line) => !(line.kind === "partial" && line.seq === msg.seq));
          filtered.push({ seq: msg.seq, text: msg.text, ts: Date.now(), kind: "final" });
          return { ...prev, [msg.project_id]: filtered.slice(-300) };
        });
        return;
      }

      if (msg.type === "error") {
        setStatusText(`Error: ${msg.message}`);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  async function startRecording() {
    if (recording) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN || !ready) {
      alert("Backend is not ready yet.");
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: true,
        autoGainControl: true,
      },
    });

    const audioContext = new AudioContext({ latencyHint: "interactive" });
    await audioContext.audioWorklet.addModule("/pcm-processor.js");

    const sourceNode = audioContext.createMediaStreamSource(stream);
    const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");

    workletNode.port.onmessage = (portEvent) => {
      const wsNow = wsRef.current;
      if (!wsNow || wsNow.readyState !== WebSocket.OPEN) return;

      const input = portEvent.data as Float32Array;
      const downsampled = downsampleFloat32(input, audioContext.sampleRate, 16000);
      const pcm16 = float32ToInt16(downsampled);
      if (pcm16.length === 0) return;

      const payload = makeBinaryAudioMessage(pcm16, {
        sampleRate: 16000,
        project_id: activeProjectIdRef.current,
        language: languageRef.current,
        session_id: sessionIdRef.current,
      });

      wsNow.send(payload);
    };

    sourceNode.connect(workletNode);
    workletNode.connect(audioContext.destination);

    streamRef.current = stream;
    audioContextRef.current = audioContext;
    sourceNodeRef.current = sourceNode;
    workletNodeRef.current = workletNode;

    seqRef.current += 1;
    ws.send(
      JSON.stringify({
        type: "set_context",
        session_id: sessionIdRef.current,
        project_id: activeProjectIdRef.current,
        language: languageRef.current,
        seq: seqRef.current,
      })
    );

    setRecording(true);
    setStatusText("Mic streaming live");
  }

  async function stopRecording() {
    if (!recording) return;

    workletNodeRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    sourceNodeRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "stop",
          session_id: sessionIdRef.current,
          project_id: activeProjectIdRef.current,
          language: languageRef.current,
          seq: seqRef.current,
        })
      );
    }

    setRecording(false);
    setStatusText("Mic stopped");
  }

  return (
    <div className="app-shell">
      <div className="layout-grid">
        <aside className="sidebar">
          <div>
            <div className="app-title">QuickCheck AI</div>
            <div className="app-subtitle">
              {connected ? "● Connected" : "○ Disconnected"} • {recording ? "🎙 Recording" : "Idle"}
            </div>
            <div className="tiny-text">{statusText}</div>
            <div className="tiny-text">Session: {sessionIdRef.current}</div>
          </div>

          <section className="panel">
            <div className="panel-label">Language</div>
            <select className="control" value={language} onChange={(e) => setLanguage(e.target.value as "auto" | "fi" | "en")}>
              <option value="auto">Auto</option>
              <option value="fi">Finnish (fi)</option>
              <option value="en">English (en)</option>
            </select>
          </section>

          <section className="panel">
            <div className="panel-label">Controls</div>
            {!recording ? (
              <button className="primary-btn" onClick={startRecording}>
                Start mic
              </button>
            ) : (
              <button className="danger-btn" onClick={stopRecording}>
                Stop mic
              </button>
            )}
            <div className="tiny-text">Streaming PCM frames through an AudioWorklet for lower latency.</div>
          </section>

          <section className="projects-wrap">
            <div className="panel-label">Projects</div>
            <div className="project-list">
              {projects.map((project) => {
                const isActive = project.id === activeProjectId;
                return (
                  <button
                    key={project.id}
                    onClick={() => setActiveProjectId(project.id)}
                    className={isActive ? "project-card active" : "project-card"}
                  >
                    <div className="project-name">{project.name}</div>
                    <div className="tiny-text">
                      {project.id} • lines: {linesByProject[project.id]?.length ?? 0}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
        </aside>

        <main className="main-panel">
          <div className="main-header">
            <div>
              <div className="panel-label">Active project</div>
              <div className="main-title">{activeProject?.name ?? activeProjectId}</div>
            </div>
            <div className="tiny-text">Realtime partials ~240 ms • final after speech end</div>
          </div>

          <section className="transcript-panel">
            <div className="transcript-header">
              <div className="project-name">Live transcript</div>
              <div className="tiny-text">Mic: {recording ? "ON" : "OFF"}</div>
            </div>

            <div className="transcript-stream">
              {activeLines.length === 0 ? (
                <div className="empty-state">Start the mic and speak. Partial text will appear while you are speaking.</div>
              ) : (
                activeLines.slice(-200).map((line, index) => (
                  <div key={`${line.seq}-${line.kind}-${index}`} className={line.kind === "final" ? "line-card final" : "line-card partial"}>
                    <div className="line-meta">
                      #{line.seq} • {line.kind.toUpperCase()} • {new Date(line.ts).toLocaleTimeString()}
                    </div>
                    <div>{line.text || <span className="tiny-text">(no text)</span>}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
