import { useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  CheckCircle, XCircle, Clock, Download, RotateCcw,
  ChevronDown, ChevronUp, AlertTriangle, FileText,
  FileCode, Braces, Save,
} from 'lucide-react'
import { api } from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface JobSummary {
  job_id:        string
  topic:         string
  mode:          string | null
  article_type:  string | null
  status:        string
  review_status: string
  created_at:    string
  finished_at:   string | null
}

interface Entities {
  organizations?: string[]
  programs?:      string[]
  locations?:     string[]
  credentials?:   string[]
  people?:        string[]
}

interface Source {
  title?: string
  slug?:  string
  score?: number
  [k: string]: unknown
}

interface Effective {
  headline:          string
  summary:           string
  body:              string
  suggested_slug:    string
  seo_summary:       string
  hashtags:          string[]
  qa_warnings:       string[]
  entities_detected: Entities
  sources_used:      Source[]
  model:             string
  input_tokens:      number | null
  output_tokens:     number | null
  generated_at:      string
}

interface JobDetail extends JobSummary {
  effective: Effective
}

type ReviewStatus = 'all' | 'pending_review' | 'approved' | 'rejected'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(iso: string): string {
  return new Date(iso + (iso.endsWith('Z') ? '' : 'Z')).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

function ReviewBadge({ status }: { status: string }) {
  const map: Record<string, { cls: string; icon: React.ReactNode; label: string }> = {
    pending_review: { cls: 'bg-yellow-100 text-yellow-700', icon: <Clock size={10} />,        label: 'Pending' },
    approved:       { cls: 'bg-green-100 text-green-700',  icon: <CheckCircle size={10} />,   label: 'Approved' },
    rejected:       { cls: 'bg-red-100 text-red-700',      icon: <XCircle size={10} />,       label: 'Rejected' },
  }
  const { cls, icon, label } = map[status] ?? map.pending_review
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${cls}`}>
      {icon}{label}
    </span>
  )
}

// ── Collapsible section ───────────────────────────────────────────────────────

function Section({ title, children, defaultOpen = false }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-50 hover:bg-gray-100 text-sm font-semibold text-gray-700"
      >
        {title}
        {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>
      {open && <div className="px-4 py-3">{children}</div>}
    </div>
  )
}

// ── Entity chips ──────────────────────────────────────────────────────────────

const ENT_CLS: Record<string, string> = {
  organizations: 'bg-purple-50 text-purple-700',
  programs:      'bg-teal-50 text-teal-700',
  locations:     'bg-orange-50 text-orange-700',
  credentials:   'bg-sky-50 text-sky-700',
  people:        'bg-pink-50 text-pink-700',
}

function EntityChips({ entities }: { entities: Entities }) {
  const entries = Object.entries(entities).filter(([, v]) => v && v.length > 0)
  if (entries.length === 0) return <p className="text-xs text-gray-400">None detected</p>
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.flatMap(([type, names]) =>
        (names ?? []).map((n: string) => (
          <span key={`${type}:${n}`}
            className={`text-[11px] px-2 py-0.5 rounded-full ${ENT_CLS[type] ?? 'bg-gray-100 text-gray-600'}`}>
            {n}
          </span>
        ))
      )}
    </div>
  )
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function DetailPanel({ job, onMutate }: {
  job:       JobDetail
  onMutate:  () => void
}) {
  const qc = useQueryClient()
  const eff = job.effective

  const [headline, setHeadline]   = useState(eff.headline)
  const [summary,  setSummary]    = useState(eff.summary)
  const [body,     setBody]       = useState(eff.body)
  const [editBody, setEditBody]   = useState(false)
  const [dirty,    setDirty]      = useState(false)

  // Reset form when job changes
  useEffect(() => {
    setHeadline(eff.headline)
    setSummary(eff.summary)
    setBody(eff.body)
    setDirty(false)
    setEditBody(false)
  }, [job.job_id])

  const mark = () => setDirty(true)

  const saveMut = useMutation({
    mutationFn: () => api.put(`/review/${job.job_id}/draft`, { headline, summary, body }),
    onSuccess: () => { setDirty(false); qc.invalidateQueries({ queryKey: ['review', job.job_id] }); onMutate() },
  })

  const approveMut = useMutation({
    mutationFn: () => api.post(`/review/${job.job_id}/approve`, {}),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['review', job.job_id] }); onMutate() },
  })

  const rejectMut = useMutation({
    mutationFn: () => api.post(`/review/${job.job_id}/reject`, {}),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['review', job.job_id] }); onMutate() },
  })

  const resetMut = useMutation({
    mutationFn: () => api.post(`/review/${job.job_id}/reset`, {}),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['review', job.job_id] }); onMutate() },
  })

  const downloadExport = (format: string) => {
    const a = document.createElement('a')
    a.href = `/api/review/${job.job_id}/export?format=${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const rs = job.review_status

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* header */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-0.5">
              <ReviewBadge status={rs} />
              {job.mode && (
                <span className="text-[11px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                  {job.mode}
                </span>
              )}
              <span className="text-xs text-gray-400">{fmtDate(job.created_at)}</span>
            </div>
            <p className="text-sm text-gray-500 truncate">{job.topic}</p>
          </div>

          {/* action buttons */}
          <div className="flex items-center gap-2 flex-shrink-0 flex-wrap justify-end">
            {dirty && (
              <button
                onClick={() => saveMut.mutate()}
                disabled={saveMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-lg"
              >
                <Save size={12} /> Save
              </button>
            )}

            {rs !== 'approved' && (
              <button
                onClick={() => approveMut.mutate()}
                disabled={approveMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs font-medium rounded-lg"
              >
                <CheckCircle size={12} /> Approve
              </button>
            )}
            {rs !== 'rejected' && (
              <button
                onClick={() => rejectMut.mutate()}
                disabled={rejectMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-lg"
              >
                <XCircle size={12} /> Reject
              </button>
            )}
            {rs !== 'pending_review' && (
              <button
                onClick={() => resetMut.mutate()}
                className="p-1.5 text-gray-400 hover:text-gray-600 border border-gray-200 rounded-lg"
                title="Reset to pending"
              >
                <RotateCcw size={13} />
              </button>
            )}

            {/* export */}
            <div className="flex items-center gap-1 border border-gray-200 rounded-lg overflow-hidden">
              <span className="px-2 text-[11px] text-gray-500 border-r border-gray-200 bg-gray-50 flex items-center gap-1 h-full py-1.5">
                <Download size={11} /> Export
              </span>
              {[
                { fmt: 'markdown', Icon: FileText,  label: 'MD'   },
                { fmt: 'html',     Icon: FileCode,  label: 'HTML' },
                { fmt: 'payload',  Icon: Braces,    label: 'JSON' },
              ].map(({ fmt, Icon, label }) => (
                <button
                  key={fmt}
                  onClick={() => downloadExport(fmt)}
                  title={`Export as ${fmt}`}
                  className="flex items-center gap-1 px-2.5 py-1.5 text-xs text-gray-600 hover:bg-gray-100 border-r last:border-r-0 border-gray-200"
                >
                  <Icon size={11} /> {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* body */}
      <div className="flex-1 overflow-auto">
        <div className="flex h-full">
          {/* main content */}
          <div className="flex-1 overflow-auto px-6 py-5 space-y-4 min-w-0">
            {/* QA warnings */}
            {eff.qa_warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-3">
                <div className="flex items-center gap-2 mb-1.5">
                  <AlertTriangle size={13} className="text-yellow-600" />
                  <p className="text-xs font-semibold text-yellow-800">QA Warnings</p>
                </div>
                <ul className="space-y-0.5">
                  {eff.qa_warnings.map((w, i) => (
                    <li key={i} className="text-xs text-yellow-700">{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Headline */}
            <div>
              <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                Headline
              </label>
              <input
                value={headline}
                onChange={e => { setHeadline(e.target.value); mark() }}
                className="w-full text-xl font-bold text-gray-900 border border-gray-200 rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Summary */}
            <div>
              <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                Summary
              </label>
              <textarea
                value={summary}
                onChange={e => { setSummary(e.target.value); mark() }}
                rows={3}
                className="w-full text-sm text-gray-700 border border-gray-200 rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
              />
            </div>

            {/* Body */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide">
                  Body
                </label>
                <button
                  onClick={() => setEditBody(e => !e)}
                  className="text-xs text-blue-500 hover:text-blue-700"
                >
                  {editBody ? 'Done editing' : 'Edit'}
                </button>
              </div>
              {editBody ? (
                <textarea
                  value={body}
                  onChange={e => { setBody(e.target.value); mark() }}
                  rows={20}
                  className="w-full text-sm font-mono text-gray-800 border border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                />
              ) : (
                <div className="bg-gray-50 border border-gray-200 rounded-xl px-5 py-4 text-sm text-gray-700 whitespace-pre-wrap leading-relaxed font-serif overflow-auto max-h-[600px]">
                  {body || <span className="text-gray-400 italic">No body content</span>}
                </div>
              )}
            </div>
          </div>

          {/* sidebar */}
          <aside className="w-72 flex-shrink-0 border-l border-gray-200 overflow-auto px-4 py-5 space-y-3 bg-gray-50">
            {/* sources */}
            <Section title={`Sources (${eff.sources_used.length})`} defaultOpen={true}>
              {eff.sources_used.length === 0 ? (
                <p className="text-xs text-gray-400">No sources</p>
              ) : (
                <ul className="space-y-2">
                  {eff.sources_used.slice(0, 8).map((s, i) => (
                    <li key={i} className="text-xs">
                      <p className="font-medium text-gray-700 leading-snug">{s.title || s.slug || `Source ${i + 1}`}</p>
                      {s.score != null && (
                        <p className="text-gray-400 tabular-nums">{Number(s.score).toFixed(3)}</p>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </Section>

            {/* entities */}
            <Section title="Entities" defaultOpen={true}>
              <EntityChips entities={eff.entities_detected} />
            </Section>

            {/* hashtags */}
            {eff.hashtags.length > 0 && (
              <Section title="Hashtags">
                <div className="flex flex-wrap gap-1">
                  {eff.hashtags.map(h => (
                    <span key={h} className="text-[11px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
                      {h}
                    </span>
                  ))}
                </div>
              </Section>
            )}

            {/* SEO */}
            {eff.seo_summary && (
              <Section title="SEO Description">
                <p className="text-xs text-gray-600 leading-relaxed">{eff.seo_summary}</p>
              </Section>
            )}

            {/* slug */}
            <Section title="Slug">
              <p className="text-xs font-mono text-gray-600 break-all">{eff.suggested_slug || '—'}</p>
            </Section>

            {/* token usage */}
            <Section title="Generation Info">
              <dl className="space-y-1 text-xs">
                {eff.model && (
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Model</dt>
                    <dd className="text-gray-700 font-mono text-[10px] truncate ml-2 max-w-[140px]">{eff.model}</dd>
                  </div>
                )}
                {eff.input_tokens != null && (
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Input tokens</dt>
                    <dd className="text-gray-700 tabular-nums">{eff.input_tokens.toLocaleString()}</dd>
                  </div>
                )}
                {eff.output_tokens != null && (
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Output tokens</dt>
                    <dd className="text-gray-700 tabular-nums">{eff.output_tokens.toLocaleString()}</dd>
                  </div>
                )}
              </dl>
            </Section>
          </aside>
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

const FILTERS: { key: ReviewStatus; label: string }[] = [
  { key: 'all',            label: 'All'      },
  { key: 'pending_review', label: 'Pending'  },
  { key: 'approved',       label: 'Approved' },
  { key: 'rejected',       label: 'Rejected' },
]

export default function ReviewQueue() {
  const qc = useQueryClient()
  const [filter,        setFilter]        = useState<ReviewStatus>('all')
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)

  const { data: jobs = [], refetch: refetchList } = useQuery<JobSummary[]>({
    queryKey: ['review-list', filter],
    queryFn:  () => {
      const qs = filter === 'all' ? '' : `?review_status=${filter}`
      return api.get<JobSummary[]>(`/review${qs}`)
    },
  })

  const { data: detail } = useQuery<JobDetail>({
    queryKey: ['review', selectedJobId],
    queryFn:  () => api.get<JobDetail>(`/review/${selectedJobId}`),
    enabled:  !!selectedJobId,
  })

  // auto-select first job when filter changes and nothing selected
  useEffect(() => {
    if (!selectedJobId && jobs.length > 0) {
      setSelectedJobId(jobs[0].job_id)
    }
  }, [jobs])

  const onMutate = () => {
    refetchList()
    qc.invalidateQueries({ queryKey: ['review-list'] })
  }

  const counts = FILTERS.map(f => ({
    ...f,
    count: f.key === 'all'
      ? jobs.length
      : jobs.filter(j => j.review_status === f.key).length,
  }))

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── left: job list ── */}
      <aside className="w-72 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col">
        <div className="px-4 pt-5 pb-3 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-sm font-bold text-gray-900 mb-3">Review Queue</h2>
          {/* filter tabs */}
          <div className="flex gap-1 flex-wrap">
            {FILTERS.map(f => {
              const cnt = f.key === 'all'
                ? jobs.length
                : jobs.filter(j => j.review_status === f.key).length
              return (
                <button
                  key={f.key}
                  onClick={() => setFilter(f.key)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                    filter === f.key
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {f.label}
                  {cnt > 0 && (
                    <span className={`ml-1 tabular-nums ${filter === f.key ? 'text-blue-200' : 'text-gray-400'}`}>
                      {cnt}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          {jobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
              <p className="text-sm text-gray-400">No drafts yet.</p>
              <p className="text-xs text-gray-400 mt-1">Generate content in the Generator page first.</p>
            </div>
          ) : (
            <ul className="divide-y divide-gray-100">
              {jobs.map(job => (
                <li key={job.job_id}>
                  <button
                    onClick={() => setSelectedJobId(job.job_id)}
                    className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                      selectedJobId === job.job_id ? 'bg-blue-50 border-r-2 border-blue-600' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <ReviewBadge status={job.review_status} />
                      <span className="text-[11px] text-gray-400 flex-shrink-0">
                        {fmtDate(job.created_at)}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-gray-800 leading-snug line-clamp-2">
                      {job.topic}
                    </p>
                    {job.mode && (
                      <p className="text-[11px] text-gray-400 mt-0.5">{job.mode}</p>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* ── right: detail ── */}
      <div className="flex-1 overflow-hidden">
        {detail ? (
          <DetailPanel key={detail.job_id} job={detail} onMutate={onMutate} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-300">
            <FileText size={36} />
            <p className="mt-3 text-sm text-gray-400">
              {jobs.length === 0
                ? 'No drafts in queue yet'
                : 'Select a draft from the left panel'}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
