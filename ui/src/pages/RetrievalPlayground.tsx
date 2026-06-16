import { useState, useRef } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Search, ChevronDown, ChevronUp, Circle } from 'lucide-react'
import { api } from '../api/client'

// ── types ────────────────────────────────────────────────────────────────────

interface Facets {
  collections: { key: string; label: string; description: string }[]
  categories:  string[]
  chunk_types: string[]
  quarters:    string[]
  years:       number[]
  languages:   string[]
}

interface ServiceStatus {
  primary:   { ready: boolean; loading: boolean }
  knowledge: { ready: boolean; loading: boolean }
}

interface Entities {
  organizations: string[]
  programs:      string[]
  locations:     string[]
  credentials:   string[]
  people:        string[]
}

interface Result {
  rank:            number
  id:              string
  score:           number
  chunk_type:      string | null
  title:           string
  slug:            string
  text:            string
  word_count:      number | null
  date:            string | null
  year:            number | null
  quarter:         string | null
  category:        string
  tags:            string[]
  source_document: string | null
  language:        string | null
  entities:        Entities
}

interface SearchResponse {
  query:      string
  collection: string
  total:      number
  elapsed_ms: number
  results:    Result[]
}

// ── small components ─────────────────────────────────────────────────────────

const SCORE_COLOUR = (s: number) =>
  s >= 0.7 ? 'bg-green-100 text-green-700'
  : s >= 0.4 ? 'bg-yellow-100 text-yellow-700'
  : 'bg-red-50 text-red-600'

const ENTITY_COLOURS: Record<keyof Entities, string> = {
  organizations: 'bg-purple-50 text-purple-700',
  programs:      'bg-teal-50 text-teal-700',
  locations:     'bg-orange-50 text-orange-700',
  credentials:   'bg-sky-50 text-sky-700',
  people:        'bg-pink-50 text-pink-700',
}

function StatusDot({ ready, loading }: { ready: boolean; loading: boolean }) {
  if (loading) return <Circle size={8} className="text-yellow-400 fill-yellow-400 animate-pulse" />
  if (ready)   return <Circle size={8} className="text-green-500 fill-green-500" />
  return             <Circle size={8} className="text-slate-600 fill-slate-600" />
}

function ResultCard({ r }: { r: Result }) {
  const [expanded, setExpanded] = useState(false)
  const TRUNCATE = 300
  const long = r.text.length > TRUNCATE

  const allEntities = (Object.keys(r.entities) as (keyof Entities)[]).filter(
    k => r.entities[k].length > 0
  )

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      {/* header row */}
      <div className="flex items-start justify-between gap-3 mb-2.5">
        <div className="flex items-center gap-2 min-w-0 flex-wrap">
          <span className="text-xs font-mono text-gray-400">#{r.rank}</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full tabular-nums ${SCORE_COLOUR(r.score)}`}>
            {r.score.toFixed(3)}
          </span>
          {r.chunk_type && (
            <span className="text-[11px] bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full font-medium">
              {r.chunk_type}
            </span>
          )}
          {r.category && (
            <span className="text-[11px] bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full">
              {r.category}
            </span>
          )}
        </div>
        <div className="text-xs text-gray-400 flex-shrink-0 tabular-nums">
          {r.word_count != null ? `${r.word_count}w` : ''}
        </div>
      </div>

      {/* title */}
      {r.title && r.title !== r.slug && (
        <p className="text-sm font-semibold text-gray-800 mb-1.5 leading-snug">{r.title}</p>
      )}

      {/* text */}
      <p className="text-sm text-gray-600 leading-relaxed">
        {expanded || !long ? r.text : r.text.slice(0, TRUNCATE) + '…'}
      </p>
      {long && (
        <button
          onClick={() => setExpanded(e => !e)}
          className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700 mt-1"
        >
          {expanded ? <><ChevronUp size={11} /> less</> : <><ChevronDown size={11} /> more</>}
        </button>
      )}

      {/* meta row */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-3 text-xs text-gray-400">
        <span className="font-mono">{r.slug}</span>
        {r.date && <span>{r.date.slice(0, 10)}</span>}
        {r.year && r.quarter && <span>{r.year} {r.quarter}</span>}
        {r.source_document && (
          <span className="truncate max-w-[200px]" title={r.source_document}>
            {r.source_document}
          </span>
        )}
      </div>

      {/* tags */}
      {r.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {r.tags.slice(0, 6).map(t => (
            <span key={t} className="text-[10px] bg-gray-50 text-gray-500 border border-gray-200 px-1.5 py-0.5 rounded">
              {t}
            </span>
          ))}
          {r.tags.length > 6 && (
            <span className="text-[10px] text-gray-400">+{r.tags.length - 6}</span>
          )}
        </div>
      )}

      {/* entities */}
      {allEntities.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-2.5 pt-2.5 border-t border-gray-50">
          {allEntities.flatMap(type =>
            r.entities[type].map(name => (
              <span key={`${type}:${name}`}
                className={`text-[11px] px-2 py-0.5 rounded-full ${ENTITY_COLOURS[type]}`}>
                {name}
              </span>
            ))
          )}
        </div>
      )}
    </div>
  )
}

// ── select helper ─────────────────────────────────────────────────────────────

function Sel({
  label, value, onChange, options, placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: string[]
  placeholder?: string
}) {
  return (
    <div>
      <label className="block text-[11px] font-medium text-gray-500 mb-1">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

function Num({
  label, value, onChange, min, max, step, placeholder,
}: {
  label: string; value: string; onChange: (v: string) => void
  min?: number; max?: number; step?: number; placeholder?: string
}) {
  return (
    <div>
      <label className="block text-[11px] font-medium text-gray-500 mb-1">{label}</label>
      <input
        type="number" min={min} max={max} step={step}
        value={value} placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        className="w-full border border-gray-200 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
    </div>
  )
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function RetrievalPlayground() {
  const [query,      setQuery]      = useState('')
  const [collection, setCollection] = useState('primary')
  const [limit,      setLimit]      = useState('10')
  const [minScore,   setMinScore]   = useState('0')
  const [year,       setYear]       = useState('')
  const [quarter,    setQuarter]    = useState('')
  const [category,   setCategory]   = useState('')
  const [chunkType,  setChunkType]  = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const facets = useQuery<Facets>({
    queryKey: ['retrieval-facets'],
    queryFn:  () => api.get<Facets>('/retrieval/facets'),
  })

  const status = useQuery<ServiceStatus>({
    queryKey:       ['retrieval-status'],
    queryFn:        () => api.get<ServiceStatus>('/retrieval/status'),
    refetchInterval: 5_000,
  })

  const searchMut = useMutation<SearchResponse, Error, void>({
    mutationFn: () =>
      api.post<SearchResponse>('/retrieval/search', {
        query,
        collection,
        limit:            parseInt(limit) || 10,
        score_threshold:  parseFloat(minScore) || 0,
        year:             year    ? parseInt(year)   : undefined,
        quarter:          quarter || undefined,
        category:         category || undefined,
        chunk_type:       chunkType || undefined,
      }),
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) searchMut.mutate()
  }

  const svc = status.data?.[collection as 'primary' | 'knowledge']

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── sticky search bar ── */}
      <div className="bg-white border-b border-gray-200 px-6 pt-5 pb-4 flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold text-gray-900">Retrieval Playground</h1>
          {svc && (
            <div className="flex items-center gap-1.5 text-xs text-gray-500">
              <StatusDot ready={svc.ready} loading={svc.loading} />
              {svc.ready ? 'Models loaded' : svc.loading ? 'Loading models…' : 'Not initialized'}
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          {/* query row */}
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Enter a search query…"
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              disabled={!query.trim() || searchMut.isPending}
              className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg disabled:opacity-50 transition-colors"
            >
              <Search size={14} />
              {searchMut.isPending ? 'Searching…' : 'Search'}
            </button>
          </div>

          {/* filter row */}
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
            {/* collection radio */}
            <div className="col-span-1">
              <label className="block text-[11px] font-medium text-gray-500 mb-1">Collection</label>
              <div className="flex flex-col gap-0.5">
                {(facets.data?.collections ?? [
                  { key: 'primary',   label: 'News' },
                  { key: 'knowledge', label: 'Knowledge' },
                ]).map(c => (
                  <label key={c.key} className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input
                      type="radio" name="collection" value={c.key}
                      checked={collection === c.key}
                      onChange={() => setCollection(c.key)}
                      className="accent-blue-600"
                    />
                    {c.label}
                  </label>
                ))}
              </div>
            </div>

            <Num label="Limit"     value={limit}    onChange={setLimit}    min={1}   max={50} placeholder="10" />
            <Num label="Min score" value={minScore}  onChange={setMinScore} min={0}   max={1}  step={0.05} placeholder="0" />
            <Num label="Year"      value={year}      onChange={setYear}                        placeholder="any" />

            <Sel label="Quarter"   value={quarter}   onChange={setQuarter}
              options={facets.data?.quarters    ?? ['Q1','Q2','Q3','Q4']}
              placeholder="any" />

            <Sel label="Chunk type" value={chunkType} onChange={setChunkType}
              options={facets.data?.chunk_types ?? ['body','summary','title']}
              placeholder="any" />
          </div>

          {/* category row */}
          {facets.data?.categories && facets.data.categories.length > 0 && (
            <div className="max-w-lg">
              <Sel label="Category" value={category} onChange={setCategory}
                options={facets.data.categories} placeholder="all categories" />
            </div>
          )}
        </form>

        {searchMut.isPending && (
          <p className="mt-2 text-xs text-gray-400">
            {!svc?.ready
              ? 'Loading embedding models (~30 s on first search)…'
              : 'Searching…'}
          </p>
        )}
      </div>

      {/* ── results ── */}
      <div className="flex-1 overflow-auto px-6 py-5">
        {searchMut.error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 mb-4">
            {searchMut.error.message}
          </div>
        )}

        {searchMut.data && (
          <>
            <p className="text-xs text-gray-400 mb-4">
              <span className="font-semibold text-gray-700">{searchMut.data.total}</span> result{searchMut.data.total !== 1 ? 's' : ''}
              {' · '}{searchMut.data.elapsed_ms} ms
              {' · '}{searchMut.data.collection === 'primary' ? 'news collection' : 'knowledge collection'}
            </p>

            {searchMut.data.results.length === 0 ? (
              <p className="text-sm text-gray-400 text-center py-12">No results. Try lowering the min score or removing filters.</p>
            ) : (
              <div className="space-y-3 max-w-4xl">
                {searchMut.data.results.map(r => (
                  <ResultCard key={r.id} r={r} />
                ))}
              </div>
            )}
          </>
        )}

        {!searchMut.data && !searchMut.error && (
          <div className="flex flex-col items-center justify-center py-20 text-gray-300">
            <Search size={36} />
            <p className="mt-3 text-sm text-gray-400">Enter a query above to search the index</p>
          </div>
        )}
      </div>
    </div>
  )
}
