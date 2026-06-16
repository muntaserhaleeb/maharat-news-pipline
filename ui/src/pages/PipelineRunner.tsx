import { useEffect, useRef, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Play, Square, AlertTriangle, CheckCircle, XCircle,
  Clock, ChevronDown, Terminal,
} from 'lucide-react'
import { api } from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Command {
  key:         string
  label:       string
  description: string
  danger:      boolean
}

interface RunSummary {
  run_id:      string
  command:     string
  label:       string
  status:      'running' | 'done' | 'error' | 'cancelled'
  started_at:  string
  finished_at: string | null
  exit_code:   number | null
  line_count:  number
}

interface RunDetail extends RunSummary {
  lines: string[]
}

// ── Line coloriser ─────────────────────────────────────────────────────────────

function lineClass(line: string): string {
  const l = line.toLowerCase()
  if (/error|exception|traceback|failed|critical/.test(l)) return 'text-red-400'
  if (/warn(?:ing)?/.test(l))                                return 'text-yellow-400'
  if (/✓|done|success|complete|finish|passed/.test(l))       return 'text-green-400'
  if (/^\[runner\]/.test(line))                              return 'text-red-400'
  if (/^\[(\d+)\/(\d+)\]|stage \d|step \d/.test(l))         return 'text-sky-400'
  if (/^(loading|embedding|upserting|verif)/.test(l))        return 'text-blue-300'
  return 'text-gray-300'
}

// ── Status badge ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: RunSummary['status'] }) {
  const map = {
    running:   { icon: <Clock size={11} className="animate-spin" />,       cls: 'bg-blue-900 text-blue-300',   label: 'running'   },
    done:      { icon: <CheckCircle size={11} />,                           cls: 'bg-green-900 text-green-300', label: 'done'      },
    error:     { icon: <XCircle size={11} />,                               cls: 'bg-red-900 text-red-300',     label: 'error'     },
    cancelled: { icon: <Square size={11} />,                                cls: 'bg-gray-700 text-gray-400',   label: 'cancelled' },
  } as const
  const { icon, cls, label } = map[status] ?? map.cancelled
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${cls}`}>
      {icon} {label}
    </span>
  )
}

// ── Relative time ─────────────────────────────────────────────────────────────

function relTime(iso: string): string {
  const diff = Math.round((Date.now() - new Date(iso + 'Z').getTime()) / 1000)
  if (diff < 60)   return `${diff}s ago`
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`
  return `${Math.round(diff / 3600)}h ago`
}

function duration(start: string, end: string | null): string {
  if (!end) return ''
  const s = Math.round((new Date(end + 'Z').getTime() - new Date(start + 'Z').getTime()) / 1000)
  if (s < 60)  return `${s}s`
  return `${Math.floor(s / 60)}m ${s % 60}s`
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function PipelineRunner() {
  const qc = useQueryClient()

  const [activeRunId,   setActiveRunId]   = useState<string | null>(null)
  const [confirmCmd,    setConfirmCmd]    = useState<Command | null>(null)
  const [showHistory,   setShowHistory]   = useState(true)
  const terminalRef = useRef<HTMLDivElement>(null)

  // ── available commands ──
  const { data: cmdsData } = useQuery<{ commands: Command[] }>({
    queryKey: ['pipeline-commands'],
    queryFn:  () => api.get<{ commands: Command[] }>('/pipeline/commands'),
    staleTime: Infinity,
  })

  // ── run history ──
  const { data: runsData, refetch: refetchRuns } = useQuery<RunSummary[]>({
    queryKey: ['pipeline-runs'],
    queryFn:  () => api.get<RunSummary[]>('/pipeline'),
  })

  const [polling, setPolling] = useState(false)

  // ── active run detail (polled while running) ──
  const { data: activeRun } = useQuery<RunDetail>({
    queryKey:        ['pipeline-run', activeRunId],
    queryFn:         () => api.get<RunDetail>(`/pipeline/${activeRunId}`),
    enabled:         !!activeRunId,
    refetchInterval: polling ? 500 : false,
  })

  // auto-scroll terminal to bottom on new lines
  useEffect(() => {
    const el = terminalRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [activeRun?.lines?.length])

  // start/stop polling and refresh history when status changes
  useEffect(() => {
    if (!activeRun) return
    const running = activeRun.status === 'running'
    setPolling(running)
    if (!running) refetchRuns()
  }, [activeRun?.status])

  // ── start run ──
  const startMut = useMutation({
    mutationFn: (command: string) =>
      api.post<{ run_id: string; status: string }>('/pipeline', { command }),
    onSuccess: (res) => {
      setActiveRunId(res.run_id)
      setPolling(true)
      qc.invalidateQueries({ queryKey: ['pipeline-runs'] })
    },
  })

  // ── cancel run ──
  const cancelMut = useMutation({
    mutationFn: (run_id: string) =>
      api.post<unknown>(`/pipeline/${run_id}/cancel`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['pipeline-run', activeRunId] })
      refetchRuns()
    },
  })

  const handleRun = (cmd: Command) => {
    if (cmd.danger) { setConfirmCmd(cmd); return }
    startMut.mutate(cmd.key)
  }

  const isRunning = activeRun?.status === 'running'
  const commands  = cmdsData?.commands ?? []
  const history   = (runsData ?? []).filter(r => r.run_id !== activeRunId)

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── left: commands ── */}
      <aside className="w-64 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col">
        <div className="px-4 py-4 border-b border-gray-200">
          <h2 className="text-sm font-bold text-gray-900">Pipeline Commands</h2>
          <p className="text-[11px] text-gray-400 mt-0.5">
            One command at a time. Runs in project venv.
          </p>
        </div>

        <div className="flex-1 overflow-auto px-3 py-3 space-y-1.5">
          {commands.map(cmd => {
            const busy = isRunning || startMut.isPending
            return (
              <button
                key={cmd.key}
                disabled={busy}
                onClick={() => handleRun(cmd)}
                title={cmd.description}
                className={`w-full text-left px-3 py-2.5 rounded-xl border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  cmd.danger
                    ? 'border-orange-200 hover:bg-orange-50 hover:border-orange-300'
                    : 'border-gray-200 hover:bg-blue-50 hover:border-blue-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  {cmd.danger && <AlertTriangle size={11} className="text-orange-500 flex-shrink-0" />}
                  <span className={`text-sm font-medium ${cmd.danger ? 'text-orange-700' : 'text-gray-800'}`}>
                    {cmd.label}
                  </span>
                </div>
                <p className="text-[11px] text-gray-400 mt-0.5 leading-snug">{cmd.description}</p>
              </button>
            )
          })}
        </div>

        {startMut.error && (
          <div className="px-3 pb-3">
            <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-2 py-1.5">
              {startMut.error.message}
            </p>
          </div>
        )}
      </aside>

      {/* ── right: terminal + history ── */}
      <div className="flex-1 flex flex-col overflow-hidden bg-gray-950">
        {/* terminal header */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-gray-900 border-b border-gray-800 flex-shrink-0">
          <div className="flex items-center gap-2">
            <Terminal size={13} className="text-gray-500" />
            {activeRun ? (
              <>
                <span className="text-sm font-medium text-gray-200">{activeRun.label}</span>
                <StatusBadge status={activeRun.status} />
                {activeRun.finished_at && (
                  <span className="text-xs text-gray-500">
                    {duration(activeRun.started_at, activeRun.finished_at)}
                  </span>
                )}
              </>
            ) : (
              <span className="text-sm text-gray-600">No run selected — click a command to start</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {isRunning && (
              <button
                onClick={() => cancelMut.mutate(activeRunId!)}
                disabled={cancelMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1 bg-red-900 hover:bg-red-800 text-red-300 text-xs font-medium rounded-lg transition-colors"
              >
                <Square size={11} /> Stop
              </button>
            )}
          </div>
        </div>

        {/* terminal body */}
        <div
          ref={terminalRef}
          className="flex-1 overflow-auto px-4 py-3 font-mono text-xs leading-5 select-text"
        >
          {activeRun ? (
            <>
              <p className="text-gray-600 mb-2">
                $ {COMMANDS_DISPLAY[activeRun.command] ?? activeRun.command}
              </p>
              {activeRun.lines.length === 0 && isRunning && (
                <span className="text-gray-600 animate-pulse">Initialising…</span>
              )}
              {activeRun.lines.map((line, i) => (
                <div key={i} className={lineClass(line)}>
                  {line || ' '}
                </div>
              ))}
              {isRunning && (
                <span className="inline-block w-2 h-3.5 bg-gray-400 animate-pulse ml-0.5 align-middle" />
              )}
              {activeRun.status === 'done' && (
                <p className="text-green-400 mt-2">
                  ✓ Process exited 0 · {duration(activeRun.started_at, activeRun.finished_at)}
                </p>
              )}
              {activeRun.status === 'error' && (
                <p className="text-red-400 mt-2">
                  ✗ Process exited {activeRun.exit_code} · {duration(activeRun.started_at, activeRun.finished_at)}
                </p>
              )}
              {activeRun.status === 'cancelled' && (
                <p className="text-yellow-400 mt-2">⊘ Cancelled</p>
              )}
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-gray-700">
              <Terminal size={32} className="mb-3" />
              <p>Select a command from the left panel</p>
            </div>
          )}
        </div>

        {/* run history */}
        {history.length > 0 && (
          <div className="border-t border-gray-800 bg-gray-900 flex-shrink-0">
            <button
              onClick={() => setShowHistory(h => !h)}
              className="w-full flex items-center justify-between px-4 py-2 text-xs text-gray-500 hover:text-gray-300"
            >
              <span>Recent runs ({history.length})</span>
              <ChevronDown size={12} className={`transition-transform ${showHistory ? 'rotate-180' : ''}`} />
            </button>

            {showHistory && (
              <div className="max-h-40 overflow-auto px-4 pb-3 space-y-1">
                {history.slice(0, 10).map(r => (
                  <button
                    key={r.run_id}
                    onClick={() => setActiveRunId(r.run_id)}
                    className={`w-full flex items-center gap-3 px-3 py-1.5 rounded-lg text-left hover:bg-gray-800 transition-colors ${
                      r.run_id === activeRunId ? 'bg-gray-800' : ''
                    }`}
                  >
                    <StatusBadge status={r.status} />
                    <span className="text-xs text-gray-300 flex-1 truncate">{r.label}</span>
                    <span className="text-[11px] text-gray-600 flex-shrink-0">
                      {relTime(r.started_at)}
                      {r.finished_at && ` · ${duration(r.started_at, r.finished_at)}`}
                    </span>
                    <span className="text-[11px] text-gray-600 flex-shrink-0">{r.line_count} lines</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── danger confirm dialog ── */}
      {confirmCmd && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-sm mx-4">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                <AlertTriangle size={18} className="text-orange-600" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-gray-900">Destructive operation</h3>
                <p className="text-xs text-gray-500">{confirmCmd.label}</p>
              </div>
            </div>
            <p className="text-sm text-gray-600 mb-5">{confirmCmd.description}</p>
            <div className="flex gap-3">
              <button
                onClick={() => {
                  startMut.mutate(confirmCmd.key)
                  setConfirmCmd(null)
                }}
                className="flex-1 py-2 bg-orange-600 hover:bg-orange-700 text-white text-sm font-semibold rounded-xl transition-colors"
              >
                Run anyway
              </button>
              <button
                onClick={() => setConfirmCmd(null)}
                className="flex-1 py-2 border border-gray-300 text-gray-700 text-sm font-medium rounded-xl hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Display strings for the terminal prompt line
const COMMANDS_DISPLAY: Record<string, string> = {
  ingest:            'python3 app/cli.py ingest',
  ingest_dry:        'python3 app/cli.py ingest --dry-run',
  rebuild:           'python3 app/cli.py rebuild-index',
  ingest_knowledge:  'python3 app/cli.py ingest-knowledge --recreate',
  evaluate:          'python3 app/cli.py evaluate --verbose',
  evaluate_knowledge:'python3 app/cli.py evaluate-knowledge --verbose',
}
