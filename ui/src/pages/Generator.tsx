import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  Wand2, Copy, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, Clock, Hash,
} from 'lucide-react'
import { api } from '../api/client'

// ── types ─────────────────────────────────────────────────────────────────────

interface Mode {
  key: string
  description: string
  word_range: { min_words?: number; max_words?: number }
}

interface ModesResponse {
  model: string
  modes: Mode[]
  article_types: string[]
}

interface Source {
  slug: string
  title: string
  date?: string
  category?: string
  chunk_type?: string
  score?: number
}

interface DraftResult {
  headline:          string
  summary:           string
  body:              string
  suggested_slug:    string
  seo_summary:       string
  hashtags:          string[]
  qa_warnings:       string[]
  entities_detected: Record<string, string[]>
  sources_used:      Source[]
  model:             string
  input_tokens:      number
  output_tokens:     number
  generated_at:      string
}

interface DraftResponse {
  job_id:     string
  dry_run:    boolean
  elapsed_ms: number
  result:     DraftResult | null
}

// ── copy helper ───────────────────────────────────────────────────────────────

function copyMarkdown(r: DraftResult) {
  const md = [
    `# ${r.headline}`,
    '',
    r.summary,
    '',
    r.body,
    '',
    '---',
    `**Slug:** ${r.suggested_slug}`,
    `**SEO summary:** ${r.seo_summary}`,
    r.hashtags.length ? `**Hashtags:** ${r.hashtags.join(' ')}` : '',
  ].filter(l => l !== undefined).join('\n')
  navigator.clipboard.writeText(md).catch(() => {})
}

// ── result panel ─────────────────────────────────────────────────────────────

function DraftPanel({ data }: { data: DraftResponse }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [copied,      setCopied]      = useState(false)
  const r = data.result

  if (data.dry_run || !r) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-5 text-sm text-yellow-700">
        Dry run mode — no generation was performed.
      </div>
    )
  }

  const handleCopy = () => {
    copyMarkdown(r)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const allEntities = Object.entries(r.entities_detected).filter(([, v]) => v.length > 0)

  return (
    <div className="space-y-5">
      {/* QA warnings */}
      {r.qa_warnings.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 space-y-1">
          {r.qa_warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2 text-sm text-yellow-700">
              <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
              {w}
            </div>
          ))}
        </div>
      )}

      {/* Article */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        {/* copy button */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100 bg-gray-50">
          <span className="text-xs text-gray-400 font-mono">{r.suggested_slug}</span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs font-medium text-gray-600 hover:text-gray-900"
          >
            {copied ? <CheckCircle2 size={13} className="text-green-500" /> : <Copy size={13} />}
            {copied ? 'Copied!' : 'Copy Markdown'}
          </button>
        </div>

        <div className="px-6 py-5 space-y-4">
          <h2 className="text-xl font-bold text-gray-900 leading-snug">{r.headline}</h2>
          <p className="text-sm text-gray-500 italic leading-relaxed">{r.summary}</p>
          <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed whitespace-pre-wrap">
            {r.body}
          </div>
        </div>
      </div>

      {/* SEO + hashtags */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
          <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-1">SEO Summary</p>
          <p className="text-sm text-gray-700">{r.seo_summary}</p>
        </div>
        {r.hashtags.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Hashtags</p>
            <div className="flex flex-wrap gap-1.5">
              {r.hashtags.map(h => (
                <span key={h} className="flex items-center gap-0.5 text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
                  <Hash size={9} />{h.replace(/^#/, '')}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Entities */}
      {allEntities.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
          <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2">Entities detected</p>
          <div className="space-y-1.5">
            {allEntities.map(([type, names]) => (
              <div key={type} className="flex items-start gap-2">
                <span className="text-xs text-gray-400 w-24 flex-shrink-0 pt-0.5 capitalize">{type}</span>
                <div className="flex flex-wrap gap-1">
                  {names.map(n => (
                    <span key={n} className="text-xs bg-purple-50 text-purple-700 px-2 py-0.5 rounded-full">{n}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sources */}
      {r.sources_used.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setSourcesOpen(o => !o)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            <span>Sources used ({r.sources_used.length})</span>
            {sourcesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {sourcesOpen && (
            <table className="w-full text-xs border-t border-gray-100">
              <thead className="bg-gray-50">
                <tr>
                  <th className="text-left px-4 py-2 text-gray-500 font-medium">Title / Slug</th>
                  <th className="text-left px-4 py-2 text-gray-500 font-medium">Date</th>
                  <th className="text-left px-4 py-2 text-gray-500 font-medium">Category</th>
                  <th className="text-left px-4 py-2 text-gray-500 font-medium">Type</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {r.sources_used.map((s, i) => (
                  <tr key={i} className="hover:bg-gray-50/50">
                    <td className="px-4 py-2 font-mono text-gray-600 max-w-[200px] truncate" title={s.title || s.slug}>
                      {s.title || s.slug}
                    </td>
                    <td className="px-4 py-2 text-gray-400">{s.date?.slice(0, 10) ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{s.category ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-400">{s.chunk_type ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Token usage + timing */}
      <div className="flex items-center gap-4 text-xs text-gray-400 px-1">
        <span className="flex items-center gap-1">
          <Clock size={11} />
          {(data.elapsed_ms / 1000).toFixed(1)} s
        </span>
        <span>{r.model}</span>
        <span>↑ {r.input_tokens.toLocaleString()} tokens</span>
        <span>↓ {r.output_tokens.toLocaleString()} tokens</span>
        <span className="font-mono text-gray-300">{data.job_id.slice(0, 8)}</span>
      </div>
    </div>
  )
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function Generator() {
  const [topic,        setTopic]        = useState('')
  const [mode,         setMode]         = useState('website_news')
  const [articleType,  setArticleType]  = useState('')
  const [useKnowledge, setUseKnowledge] = useState(false)
  const [year,         setYear]         = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [limit,        setLimit]        = useState('')
  const [scoreThresh,  setScoreThresh]  = useState('')

  const modesQ = useQuery<ModesResponse>({
    queryKey: ['generator-modes'],
    queryFn:  () => api.get<ModesResponse>('/generator/modes'),
  })

  const draftMut = useMutation<DraftResponse, Error, void>({
    mutationFn: () =>
      api.post<DraftResponse>('/generator/draft', {
        topic,
        mode,
        article_type:    articleType || undefined,
        use_knowledge:   useKnowledge,
        dry_run:         false,
        year:            year       ? parseInt(year)       : undefined,
        limit:           limit      ? parseInt(limit)      : undefined,
        score_threshold: scoreThresh ? parseFloat(scoreThresh) : undefined,
      }),
  })

  const selectedMode = modesQ.data?.modes.find(m => m.key === mode)

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── left: form ── */}
      <aside className="w-72 bg-white border-r border-gray-200 flex flex-col flex-shrink-0 overflow-auto">
        <div className="px-5 py-4 border-b border-gray-100">
          <h1 className="text-base font-bold text-gray-900 flex items-center gap-2">
            <Wand2 size={15} className="text-blue-500" /> Content Generator
          </h1>
          {modesQ.data?.model && (
            <p className="text-xs text-gray-400 mt-0.5 font-mono">{modesQ.data.model}</p>
          )}
        </div>

        <div className="flex-1 px-5 py-4 space-y-4">
          {/* topic */}
          <div>
            <label className="block text-xs font-semibold text-gray-600 mb-1.5">Topic *</label>
            <textarea
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="e.g. Maharat and Samsung partnership graduation ceremony"
              rows={3}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* mode */}
          <div>
            <label className="block text-xs font-semibold text-gray-600 mb-1.5">Generation mode</label>
            <select
              value={mode}
              onChange={e => setMode(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {(modesQ.data?.modes ?? []).map(m => (
                <option key={m.key} value={m.key}>{m.key}</option>
              ))}
            </select>
            {selectedMode && (
              <p className="text-xs text-gray-400 mt-1">{selectedMode.description}</p>
            )}
            {selectedMode?.word_range?.min_words && (
              <p className="text-[11px] text-gray-400">
                {selectedMode.word_range.min_words}–{selectedMode.word_range.max_words} words
              </p>
            )}
          </div>

          {/* options */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useKnowledge}
                onChange={e => setUseKnowledge(e.target.checked)}
                className="accent-blue-600 w-3.5 h-3.5"
              />
              <span className="text-sm text-gray-700">Use knowledge base</span>
            </label>
          </div>

          {/* year */}
          <div>
            <label className="block text-xs font-semibold text-gray-600 mb-1.5">Year filter</label>
            <input
              type="number"
              value={year}
              onChange={e => setYear(e.target.value)}
              placeholder="e.g. 2026  (optional)"
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* advanced toggle */}
          <button
            onClick={() => setShowAdvanced(o => !o)}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
          >
            {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            Advanced options
          </button>

          {showAdvanced && (
            <div className="space-y-3 border-t border-gray-100 pt-3">
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Article type</label>
                <select
                  value={articleType}
                  onChange={e => setArticleType(e.target.value)}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">auto</option>
                  {(modesQ.data?.article_types ?? []).map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Context chunks limit</label>
                <input
                  type="number" min={1} max={20}
                  value={limit}
                  onChange={e => setLimit(e.target.value)}
                  placeholder="default (8)"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-500 mb-1">Score threshold</label>
                <input
                  type="number" min={0} max={1} step={0.05}
                  value={scoreThresh}
                  onChange={e => setScoreThresh(e.target.value)}
                  placeholder="default (0.15)"
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          )}
        </div>

        {/* generate button */}
        <div className="px-5 py-4 border-t border-gray-100">
          <button
            onClick={() => { if (topic.trim()) draftMut.mutate() }}
            disabled={!topic.trim() || draftMut.isPending}
            className="w-full flex items-center justify-center gap-2 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg disabled:opacity-50 transition-colors"
          >
            <Wand2 size={14} />
            {draftMut.isPending ? 'Generating…' : 'Generate'}
          </button>
          {draftMut.isPending && (
            <p className="text-[11px] text-gray-400 text-center mt-2">
              Retrieval + Claude · may take 30–90 s
            </p>
          )}
        </div>
      </aside>

      {/* ── right: output ── */}
      <div className="flex-1 overflow-auto px-6 py-6">
        {draftMut.error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
            <strong>Error:</strong> {draftMut.error.message}
          </div>
        )}

        {draftMut.data && <DraftPanel data={draftMut.data} />}

        {!draftMut.data && !draftMut.error && (
          <div className="flex flex-col items-center justify-center h-full text-gray-300 gap-3">
            <Wand2 size={40} />
            <p className="text-sm text-gray-400">Fill in the form and click Generate</p>
            <p className="text-xs text-gray-300 text-center max-w-xs">
              Uses hybrid Qdrant retrieval + Claude to draft a grounded article.
              Requires <code className="font-mono text-xs">ANTHROPIC_API_KEY</code> in the shell environment.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
