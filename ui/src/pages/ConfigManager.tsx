import { useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Save, RotateCcw, Clock, FileCode2, ShieldCheck, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { api } from '../api/client'

interface ConfigFile   { name: string; exists: boolean; size: number }
interface ConfigDetail { name: string; content: string; parsed: Record<string, unknown> | null }
interface Version      { id: number; config_name: string; saved_at: string; note: string | null }
interface ValidateResult { valid: boolean; errors: string[]; warnings: string[] }

type Tab = 'edit' | 'history'

function ValidationPanel({ result }: { result: ValidateResult }) {
  if (result.valid && result.warnings.length === 0) {
    return (
      <div className="flex items-center gap-2 px-4 py-2.5 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700">
        <CheckCircle2 size={14} />
        <span className="font-medium">Valid YAML</span>
        <span className="text-green-500">— no errors or warnings</span>
      </div>
    )
  }
  return (
    <div className="space-y-2">
      {result.errors.map((e, i) => (
        <div key={i} className="flex items-start gap-2 px-4 py-2.5 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          <XCircle size={14} className="flex-shrink-0 mt-0.5" />
          <pre className="whitespace-pre-wrap font-mono text-xs">{e}</pre>
        </div>
      ))}
      {result.warnings.map((w, i) => (
        <div key={i} className="flex items-start gap-2 px-4 py-2.5 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-700">
          <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
          <span>{w}</span>
        </div>
      ))}
    </div>
  )
}

export default function ConfigManager() {
  const [selected,      setSelected]      = useState<string | null>(null)
  const [tab,           setTab]           = useState<Tab>('edit')
  const [editorValue,   setEditorValue]   = useState('')
  const [note,          setNote]          = useState('')
  const [saveFeedback,  setSaveFeedback]  = useState<{ ok: boolean; msg: string } | null>(null)
  const [validateResult, setValidateResult] = useState<ValidateResult | null>(null)
  const qc = useQueryClient()

  const { data: configs } = useQuery<ConfigFile[]>({
    queryKey: ['config-list'],
    queryFn:  () => api.get<ConfigFile[]>('/config'),
  })

  const { data: detail, isLoading: detailLoading } = useQuery<ConfigDetail>({
    queryKey: ['config-detail', selected],
    queryFn:  () => api.get<ConfigDetail>(`/config/${selected}`),
    enabled:  !!selected,
  })

  const { data: history } = useQuery<Version[]>({
    queryKey: ['config-history', selected],
    queryFn:  () => api.get<Version[]>(`/config/${selected}/history`),
    enabled:  !!selected && tab === 'history',
  })

  // Sync editor when a different file is loaded
  useEffect(() => {
    if (detail) {
      setEditorValue(detail.content)
      setValidateResult(null)
    }
  }, [detail])

  // Clear validation result on any edit
  const handleEditorChange = (val: string) => {
    setEditorValue(val)
    setValidateResult(null)
  }

  const validateMutation = useMutation<ValidateResult, Error, void>({
    mutationFn: () =>
      api.post<ValidateResult>('/config/validate', {
        content: editorValue,
        name: selected,
      }),
    onSuccess: (result) => setValidateResult(result),
    onError:   (e)      => setValidateResult({ valid: false, errors: [e.message], warnings: [] }),
  })

  const saveMutation = useMutation<unknown, Error, void>({
    mutationFn: () =>
      api.put<unknown>(`/config/${selected}`, { content: editorValue, note: note || null }),
    onSuccess: () => {
      setSaveFeedback({ ok: true, msg: 'Saved successfully' })
      setNote('')
      qc.invalidateQueries({ queryKey: ['config-history', selected] })
      setTimeout(() => setSaveFeedback(null), 3000)
    },
    onError: (e) => setSaveFeedback({ ok: false, msg: e.message }),
  })

  const rollbackMutation = useMutation<unknown, Error, number>({
    mutationFn: (id) => api.post<unknown>(`/config/${selected}/rollback/${id}`, {}),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['config-detail',  selected] })
      qc.invalidateQueries({ queryKey: ['config-history', selected] })
      setTab('edit')
      setSaveFeedback({ ok: true, msg: 'Rolled back successfully' })
      setTimeout(() => setSaveFeedback(null), 3000)
    },
    onError: (e) => setSaveFeedback({ ok: false, msg: e.message }),
  })

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── File list ── */}
      <aside className="w-56 border-r border-gray-200 bg-white flex flex-col flex-shrink-0">
        <div className="px-4 py-3 border-b border-gray-100">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Config Files</p>
        </div>
        <div className="flex-1 overflow-auto py-1.5">
          {configs?.map((c) => (
            <button
              key={c.name}
              onClick={() => { setSelected(c.name); setTab('edit') }}
              className={`w-full text-left flex items-center gap-2 px-4 py-2.5 text-sm transition-colors ${
                selected === c.name
                  ? 'bg-blue-50 text-blue-700 font-medium border-r-2 border-blue-600'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <FileCode2 size={13} className="flex-shrink-0 opacity-40" />
              <span className="truncate">{c.name}</span>
              {!c.exists && (
                <span className="text-[10px] text-red-400 ml-auto flex-shrink-0">missing</span>
              )}
            </button>
          ))}
        </div>
      </aside>

      {/* ── Editor panel ── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {!selected ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 text-gray-300">
            <FileCode2 size={32} />
            <p className="text-sm text-gray-400">Select a config file to view or edit</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center gap-3 flex-shrink-0">
              <span className="text-sm font-semibold text-gray-800">{selected}</span>
              <div className="ml-auto flex gap-1">
                {(['edit', 'history'] as Tab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded font-medium ${
                      tab === t ? 'bg-blue-600 text-white' : 'text-gray-500 hover:bg-gray-100'
                    }`}
                  >
                    {t === 'edit' ? <><FileCode2 size={11} /> Edit</> : <><Clock size={11} /> History</>}
                  </button>
                ))}
              </div>
            </div>

            {/* Save feedback bar */}
            {saveFeedback && (
              <div className={`px-6 py-2 text-xs font-medium flex-shrink-0 ${
                saveFeedback.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
              }`}>
                {saveFeedback.msg}
              </div>
            )}

            {/* ── Edit tab ── */}
            {tab === 'edit' && (
              <div className="flex-1 flex flex-col overflow-hidden p-5 gap-3">
                {detailLoading ? (
                  <div className="text-sm text-gray-400">Loading…</div>
                ) : (
                  <>
                    <textarea
                      className="flex-1 font-mono text-xs bg-gray-50 border border-gray-200 rounded-lg p-4 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent leading-relaxed"
                      value={editorValue}
                      onChange={(e) => handleEditorChange(e.target.value)}
                      spellCheck={false}
                    />

                    {/* Validation result */}
                    {validateResult && <ValidationPanel result={validateResult} />}

                    {/* Action row */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <button
                        onClick={() => validateMutation.mutate()}
                        disabled={validateMutation.isPending}
                        className="flex items-center gap-1.5 px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 text-xs font-semibold rounded-lg disabled:opacity-50 transition-colors"
                      >
                        <ShieldCheck size={13} />
                        {validateMutation.isPending ? 'Validating…' : 'Validate'}
                      </button>

                      <input
                        type="text"
                        placeholder="Version note (optional)…"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        className="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />

                      <button
                        onClick={() => saveMutation.mutate()}
                        disabled={saveMutation.isPending}
                        className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg disabled:opacity-50 transition-colors"
                      >
                        <Save size={13} />
                        {saveMutation.isPending ? 'Saving…' : 'Save'}
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* ── History tab ── */}
            {tab === 'history' && (
              <div className="flex-1 overflow-auto p-5">
                {!history?.length ? (
                  <p className="text-sm text-gray-400">
                    No version history yet. Save a change to start tracking.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {history.map((v) => (
                      <div key={v.id} className="bg-white border border-gray-200 rounded-lg px-4 py-3 flex items-center gap-4">
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-mono text-gray-700">
                            {new Date(v.saved_at + 'Z').toLocaleString()}
                          </p>
                          {v.note && (
                            <p className="text-xs text-gray-400 mt-0.5 truncate">{v.note}</p>
                          )}
                        </div>
                        <button
                          onClick={() => rollbackMutation.mutate(v.id)}
                          disabled={rollbackMutation.isPending}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-gray-300 rounded-lg hover:bg-gray-50 text-gray-600 disabled:opacity-50 transition-colors flex-shrink-0"
                        >
                          <RotateCcw size={11} /> Restore
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
