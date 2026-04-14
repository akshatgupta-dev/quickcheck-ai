import "./App.css";
import { useEffect, useMemo, useRef, useState } from "react";

type Project = { id: string; name: string };

type TranscriptLine = { seq: number; text: string; ts: number; kind: "partial" | "final" };

type ProjectSummary = {
  session_id: string;
  project_id: string;
  bullets: string[];
  transcript: string;
  line_count: number;
  word_count: number;
  generated_at: number;
};

type SessionSummary = {
  session_id: string;
  projects: ProjectSummary[];
};

type ReadyMsg = {
  type: "ready";
  session_id: string;
  project_id: string;
  language: string;
  sample_rate: number;
  frame_ms: number;
};

type PartialMsg = {
  type: "partial";
  session_id: string;
  project_id: string;
  text: string;
  seq: number;
  is_final: false;
};

type FinalMsg = {
  type: "final";
  session_id: string;
  project_id: string;
  text: string;
  seq: number;
  is_final: true;
};

type ProjectTranscriptMsg = {
  type: "project_transcript";
  session_id: string;
  project_id: string;
  text: string;
  line_count: number;
};

type ProjectSummaryMsg = ProjectSummary & { type: "project_summary" };

type SessionSummaryMsg = {
  type: "session_summary";
  session_id: string;
  projects: ProjectSummary[];
};

type StoppedMsg = {
  type: "stopped";
  session_id: string;
};

type ErrorMsg = {
  type: "error";
  message: string;
  project_id?: string;
  seq?: number;
};

type WsMsg =
  | ReadyMsg
  | PartialMsg
  | FinalMsg
  | ProjectTranscriptMsg
  | ProjectSummaryMsg
  | SessionSummaryMsg
  | StoppedMsg
  | ErrorMsg
  | { type: "pong" };

const DEFAULT_PROJECTS: Project[] = [
  { id: "proj-1", name: "Client interview" },
  { id: "proj-2", name: "Product planning" },
  { id: "proj-3", name: "Technical notes" },
];

function makeWsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const host = location.hostname || "localhost";
  return `${proto}://${host}:8000/ws/transcribe`;
}

function downsampleFloat32(input: Float32Array, inputSampleRate: number, targetSampleRate: number): Float32Array {
  if (input.length === 0) {
    return input;
  }

  if (targetSampleRate >= inputSampleRate) {
    return input;
  }

  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.max(1, Math.round(input.length / ratio));
  const result = new Float32Array(newLength);

  let offsetResult = 0;
  let offsetInput = 0;

  while (offsetResult < result.length) {
    const nextOffsetInput = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;

    for (let index = offsetInput; index < Math.min(nextOffsetInput, input.length); index += 1) {
      accum += input[index];
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
  for (let index = 0; index < float32.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32[index]));
    int16[index] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return int16;
}

function makeBinaryAudioMessage(audioBuffer: Int16Array, metadata: Record<string, unknown>): ArrayBuffer {
  const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
  const output = new Uint8Array(4 + metadataBytes.byteLength + audioBuffer.byteLength);
  new DataView(output.buffer).setUint32(0, metadataBytes.byteLength, true);
  output.set(metadataBytes, 4);
  output.set(new Uint8Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength), 4 + metadataBytes.byteLength);
  return output.buffer;
}

function makeProjectId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return `proj-${cryptoApi.randomUUID()}`;
  }
  return `proj-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>(DEFAULT_PROJECTS);
  const [newProjectName, setNewProjectName] = useState("");
  const [activeProjectId, setActiveProjectId] = useState(DEFAULT_PROJECTS[0]?.id ?? "proj-1");
  const [language, setLanguage] = useState<"auto" | "fi" | "en">("auto");
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [statusText, setStatusText] = useState("Waiting for backend connection");
  const [linesByProject, setLinesByProject] = useState<Record<string, TranscriptLine[]>>({});
  const [summaryByProject, setSummaryByProject] = useState<Record<string, ProjectSummary>>({});
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef<string>(`session-${Date.now()}`);
  const activeProjectIdRef = useRef(activeProjectId);
  const languageRef = useRef(language);
  const packetSeqRef = useRef(0);
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);

  const activeProject = useMemo(() => projects.find((project) => project.id === activeProjectId), [projects, activeProjectId]);
  const activeLines = linesByProject[activeProjectId] ?? [];
  const activeSummary = summaryByProject[activeProjectId];

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
    languageRef.current = language;

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "set_context",
          session_id: sessionIdRef.current,
          project_id: activeProjectIdRef.current,
          language: languageRef.current,
        })
      );
    }
  }, [activeProjectId, language]);

  useEffect(() => {
    const element = transcriptScrollRef.current;
    if (!element) {
      return;
    }

    element.scrollTop = element.scrollHeight;
  }, [activeProjectId, activeLines.length]);

  useEffect(() => {
    const ws = new WebSocket(makeWsUrl());
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setReady(false);
      setStatusText("Backend connected");

      ws.send(
        JSON.stringify({
          type: "config",
          session_id: sessionIdRef.current,
          project_id: activeProjectIdRef.current,
          language: languageRef.current,
          sample_rate: 16000,
        })
      );
    };

    ws.onclose = () => {
      setConnected(false);
      setReady(false);
      setStatusText("Backend disconnected");
    };

    ws.onerror = () => {
      setStatusText("Backend connection error");
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") {
        return;
      }

      try {
        const msg = JSON.parse(event.data) as WsMsg;

        if (msg.type === "ready") {
          setReady(true);
          setStatusText(`Ready • ${msg.sample_rate} Hz • ${msg.frame_ms} ms frames`);
          return;
        }

        if (msg.type === "partial" || msg.type === "final") {
          setLinesByProject((prev) => {
            const current = prev[msg.project_id] ? [...prev[msg.project_id]] : [];
            const filtered = current.filter((line) => line.seq !== msg.seq);
            filtered.push({ seq: msg.seq, text: msg.text, ts: Date.now(), kind: msg.type });
            return { ...prev, [msg.project_id]: filtered.slice(-300) };
          });
          return;
        }

        if (msg.type === "project_summary") {
          setSummaryByProject((prev) => ({ ...prev, [msg.project_id]: msg }));
          setStatusText(`Summary ready for ${msg.project_id}`);
          return;
        }

        if (msg.type === "session_summary") {
          setSessionSummary({ session_id: msg.session_id, projects: msg.projects });
          setSummaryByProject((prev) => {
            const next = { ...prev };
            for (const projectSummary of msg.projects) {
              next[projectSummary.project_id] = projectSummary;
            }
            return next;
          });
          setStatusText(`Session summary updated • ${msg.projects.length} project${msg.projects.length === 1 ? "" : "s"}`);
          return;
        }

        if (msg.type === "stopped") {
          setStatusText("Mic stopped");
          return;
        }

        if (msg.type === "error") {
          setStatusText(`Error: ${msg.message}`);
        }
      } catch {
        setStatusText("Received malformed backend response");
      }
    };

    return () => {
      ws.close();
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, []);

  function sendControlMessage(payload: Record<string, unknown>): boolean {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    ws.send(JSON.stringify(payload));
    return true;
  }

  function createProject() {
    const name = newProjectName.trim();
    if (!name) {
      return;
    }

    const id = makeProjectId();
    setProjects((prev) => [...prev, { id, name }]);
    setNewProjectName("");
    setActiveProjectId(id);
    setStatusText(`Created project ${name}`);
  }

  function requestActiveProjectSummary() {
    const projectId = activeProjectIdRef.current;
    const sent = sendControlMessage({
      type: "get_project_summary",
      session_id: sessionIdRef.current,
      project_id: projectId,
    });

    if (!sent) {
      setStatusText("Backend is not connected");
      return;
    }

    setStatusText(`Refreshing summary for ${projects.find((project) => project.id === projectId)?.name ?? projectId}`);
  }

  function requestSessionSummary() {
    const sent = sendControlMessage({
      type: "get_session_summary",
      session_id: sessionIdRef.current,
    });

    if (!sent) {
      setStatusText("Backend is not connected");
      return;
    }

    setStatusText("Refreshing session summary");
  }

  async function startRecording() {
    if (recording) {
      return;
    }

    if (!connected || !ready) {
      setStatusText("Wait for backend readiness before starting the mic");
      return;
    }

    try {
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
      await audioContext.resume();

      const sourceNode = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "pcm-processor");
      const gainNode = audioContext.createGain();
      gainNode.gain.value = 0;

      packetSeqRef.current = 0;

      workletNode.port.onmessage = (event) => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          return;
        }

        const input = event.data as Float32Array;
        if (!(input instanceof Float32Array) || input.length === 0) {
          return;
        }

        const downsampled = downsampleFloat32(input, audioContext.sampleRate, 16000);
        const pcm16 = float32ToInt16(downsampled);
        if (pcm16.length === 0) {
          return;
        }

        packetSeqRef.current += 1;
        ws.send(
          makeBinaryAudioMessage(pcm16, {
            session_id: sessionIdRef.current,
            project_id: activeProjectIdRef.current,
            language: languageRef.current,
            sampleRate: 16000,
            seq: packetSeqRef.current,
          })
        );
      };

      sourceNode.connect(workletNode);
      workletNode.connect(gainNode);
      gainNode.connect(audioContext.destination);

      streamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceNodeRef.current = sourceNode;
      workletNodeRef.current = workletNode;
      gainNodeRef.current = gainNode;

      const sentContext = sendControlMessage({
        type: "set_context",
        session_id: sessionIdRef.current,
        project_id: activeProjectIdRef.current,
        language: languageRef.current,
      });

      setRecording(true);
      setStatusText(sentContext ? "Mic streaming live" : "Mic streaming live • backend context pending");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to start microphone";
      setStatusText(message);
    }
  }

  async function stopRecording() {
    if (!recording && !streamRef.current) {
      return;
    }

    try {
      workletNodeRef.current?.disconnect();
      sourceNodeRef.current?.disconnect();
      gainNodeRef.current?.disconnect();
    } catch {
      // Ignore cleanup errors.
    }

    workletNodeRef.current = null;
    sourceNodeRef.current = null;
    gainNodeRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setRecording(false);

    sendControlMessage({
      type: "stop",
      session_id: sessionIdRef.current,
      project_id: activeProjectIdRef.current,
      language: languageRef.current,
    });

    requestActiveProjectSummary();
    requestSessionSummary();
    setStatusText("Mic stopped • summaries requested");
  }

  const sessionProjectCount = sessionSummary?.projects.length ?? 0;
  const summaryProjectCount = Object.keys(summaryByProject).length;

  return (
    <div className="app-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <div className="shell-grid">
        <aside className="sidebar">
          <div className="brand-row">
            <div className="brand-mark">Q</div>
            <div>
              <div className="eyebrow">QuickCheck AI</div>
              <div className="app-title">Transcribe, review, summarize</div>
            </div>
          </div>

          <p className="app-copy">
            Capture a conversation once, keep it organized by project, and finish with bullet-point summaries you can use immediately.
          </p>

          <div className="status-stack">
            <span className={`status-chip ${connected ? "good" : "bad"}`}>
              {connected ? "Backend connected" : "Backend offline"}
            </span>
            <span className={`status-chip ${ready ? "good" : "warn"}`}>
              {ready ? "Stream ready" : "Waiting for ready"}
            </span>
            <span className={`status-chip ${recording ? "info" : "soft"}`}>
              {recording ? "Recording live" : "Idle"}
            </span>
          </div>

          <div className="card">
            <div className="card-head">
              <div>
                <div className="card-title">Session</div>
                <div className="card-subtitle">{statusText}</div>
              </div>
            </div>

            <div className="detail-row">
              <span className="card-subtitle">Session ID</span>
              <span className="mono">{sessionIdRef.current}</span>
            </div>
            <div className="detail-row">
              <span className="card-subtitle">Active project</span>
              <span>{activeProject?.name ?? activeProjectId}</span>
            </div>
            <div className="detail-row">
              <span className="card-subtitle">Summaries ready</span>
              <span>{summaryProjectCount}</span>
            </div>
          </div>

          <div className="card">
            <div className="field-label">Language</div>
            <select className="select" value={language} onChange={(event) => setLanguage(event.target.value as "auto" | "fi" | "en")}>
              <option value="auto">Auto</option>
              <option value="fi">Finnish (fi)</option>
              <option value="en">English (en)</option>
            </select>
            <div className="helper-text" style={{ marginTop: 10 }}>
              Change this anytime. The backend applies the setting to the next spoken turn.
            </div>
          </div>

          <div className="card">
            <div className="field-label">Controls</div>
            <div className="button-row">
              {!recording ? (
                <button className="primary-button" onClick={() => void startRecording()} disabled={!connected || !ready}>
                  Start mic
                </button>
              ) : (
                <button className="danger-button" onClick={() => void stopRecording()}>
                  Stop mic
                </button>
              )}
              <button className="secondary-button" onClick={requestActiveProjectSummary} disabled={!connected}>
                Summarize active project
              </button>
              <button className="ghost-button" onClick={requestSessionSummary} disabled={!connected}>
                Summarize all projects
              </button>
            </div>
            <div className="helper-text" style={{ marginTop: 10 }}>
              Switch projects while recording. The next speech turn will be tagged correctly, and the old turn will flush cleanly.
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <div>
                <div className="card-title">Projects</div>
                <div className="card-subtitle">{projects.length} in this session</div>
              </div>
            </div>

            <div className="project-create">
              <input
                className="input"
                value={newProjectName}
                onChange={(event) => setNewProjectName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    createProject();
                  }
                }}
                placeholder="Add a project"
              />
              <button className="secondary-button" onClick={createProject}>
                Add
              </button>
            </div>

            <div className="project-list" style={{ marginTop: 12 }}>
              {projects.map((project) => {
                const isActive = project.id === activeProjectId;
                const summary = summaryByProject[project.id];
                const lineCount = linesByProject[project.id]?.length ?? 0;

                return (
                  <button key={project.id} onClick={() => setActiveProjectId(project.id)} className={`project-card ${isActive ? "active" : ""}`}>
                    <div className="project-card-top">
                      <div>
                        <div className="project-name">{project.name}</div>
                        <div className="project-meta">
                          {project.id} • {lineCount} line{lineCount === 1 ? "" : "s"}
                        </div>
                      </div>
                      <span className={`status-chip ${summary ? "good" : "soft"}`}>{summary ? `${summary.bullets.length} bullets` : "Pending"}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="main-panel">
          <section className="hero">
            <div>
              <div className="eyebrow">Live capture</div>
              <h1>{activeProject?.name ?? activeProjectId}</h1>
              <p>{recording ? "Streaming audio into the backend and splitting it into speech turns." : "Start the mic to capture speech turns for this project."}</p>
            </div>

            <div className="hero-metrics">
              <div className="hero-stat">
                <div className="metric-label">Connection</div>
                <div className="hero-stat-value">{connected ? "Online" : "Offline"}</div>
              </div>
              <div className="hero-stat">
                <div className="metric-label">Mode</div>
                <div className="hero-stat-value">{recording ? "Recording" : "Idle"}</div>
              </div>
              <div className="hero-stat">
                <div className="metric-label">Active lines</div>
                <div className="hero-stat-value">{activeLines.length}</div>
              </div>
              <div className="hero-stat">
                <div className="metric-label">Bullet summaries</div>
                <div className="hero-stat-value">{summaryProjectCount}</div>
              </div>
            </div>
          </section>

          <section className="metric-grid">
            <article className="metric-card">
              <div className="metric-label">Session</div>
              <div className="metric-value">{sessionIdRef.current.slice(0, 20)}…</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Projects</div>
              <div className="metric-value">{projects.length}</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Summarized projects</div>
              <div className="metric-value">{sessionProjectCount}</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Transcript status</div>
              <div className="metric-value">{activeSummary ? `${activeSummary.line_count} lines` : "Waiting"}</div>
            </article>
          </section>

          <section className="content-grid">
            <div className="panel transcript-panel">
              <div className="panel-header">
                <div>
                  <div className="panel-title">Live transcript</div>
                  <div className="panel-copy">Partial lines are replaced with final lines when the speech turn closes.</div>
                </div>
                <span className="status-chip soft">Mic: {recording ? "ON" : "OFF"}</span>
              </div>

              <div className="transcript-list" ref={transcriptScrollRef}>
                {activeLines.length === 0 ? (
                  <div className="transcript-empty">No transcript yet. Speak once the mic is live.</div>
                ) : (
                  activeLines.slice(-200).map((line) => (
                    <article key={`${line.seq}-${line.kind}`} className={`transcript-line ${line.kind}`}>
                      <div className="line-meta">
                        #{line.seq} • {line.kind.toUpperCase()} • {new Date(line.ts).toLocaleTimeString()}
                      </div>
                      <div className="transcript-line-body">{line.text || <span className="empty-state">(no text)</span>}</div>
                    </article>
                  ))
                )}
              </div>
            </div>

            <div className="panel summary-panel">
              <div className="panel-header">
                <div>
                  <div className="panel-title">Project summaries</div>
                  <div className="panel-copy">
                    {sessionProjectCount > 0
                      ? `${sessionProjectCount} project summary${sessionProjectCount === 1 ? "" : "ies"} captured in this session.`
                      : "Summaries appear here after you stop recording or refresh them manually."}
                  </div>
                </div>
                <span className="status-chip soft">{summaryProjectCount} ready</span>
              </div>

              <div className="summary-grid">
                {projects.map((project) => {
                  const summary = summaryByProject[project.id];
                  const isActive = project.id === activeProjectId;

                  return (
                    <article key={project.id} className={`summary-card ${isActive ? "active" : ""}`}>
                      <div className="summary-head">
                        <div>
                          <div className="project-name">{project.name}</div>
                          <div className="project-meta">
                            {summary ? `${summary.line_count} lines • ${summary.word_count} words` : "Waiting for a summary"}
                          </div>
                        </div>
                        <span className={`status-chip ${summary ? "good" : "soft"}`}>
                          {summary ? `${summary.bullets.length} bullets` : "Pending"}
                        </span>
                      </div>

                      {summary ? (
                        <ul className="summary-bullets">
                          {summary.bullets.map((bullet) => (
                            <li key={bullet}>{bullet}</li>
                          ))}
                        </ul>
                      ) : (
                        <div className="summary-empty">Stop the mic or click refresh to turn this project transcript into a bullet summary.</div>
                      )}
                    </article>
                  );
                })}
              </div>

              <div className="summary-footer">
                <span className="status-chip good">{sessionSummary ? "Session summary ready" : "Session summary pending"}</span>
                <span className="panel-copy">
                  {sessionSummary ? `Last updated ${new Date(sessionSummary.projects[0]?.generated_at ?? Date.now()).toLocaleTimeString()}` : "Captions, transcripts, and bullets stay grouped by project."}
                </span>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}