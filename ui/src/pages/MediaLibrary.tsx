import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FolderOpen, ScanSearch, Sparkles, CheckCircle, Download,
  X, ChevronDown, ChevronRight, AlertCircle, Clock,
  Image as ImageIcon, Star, Loader2, RefreshCw,
} from 'lucide-react'
import { api } from '../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface EventSummary {
  event_id:      string
  folder_name:   string
  event_name:    string
  event_date:    string | null
  status:        string
  image_count:   number
  hero:          string | null
  gallery_count: number
}

interface ImageScore {
  total:      number
  resolution: number
  sharpness:  number
  exposure:   number
  filesize:   number
  width:      number
  height:     number
  bytes:      number
  error?:     string
}

interface AIResult {
  subjects?:         string[]
  composition_score?: number
  hero_confidence?:  number
  hero_reason?:      string
  alt_text_en?:      string
  alt_text_ar?:      string
  caption_en?:       string
  caption_ar?:       string
  tags?:             string[]
  error?:            string
}

interface AIRanking {
  recommended_hero?: string
  ranking?:          string[]
  reason?:           string
}

interface EventDetail extends EventSummary {
  base_dir:  string
  scores:    Record<string, ImageScore>
  duplicates: string[][]
  gallery:   string[]
  rejected:  Record<string, string>
  metadata:  Record<string, string | string[]>
  ai:        Record<string, AIResult | AIRanking>
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  pending:     'bg-gray-100 text-gray-500',
  scanning:    'bg-yellow-100 text-yellow-700',
  analyzing:   'bg-purple-100 text-purple-700',
  scored:      'bg-blue-100 text-blue-700',
  ai_analyzed: 'bg-indigo-100 text-indigo-700',
  approved:    'bg-green-100 text-green-700',
  error:       'bg-red-100 text-red-700',
}

const STATUS_ICON: Record<string, JSX.Element> = {
  pending:     <Clock size={12} />,
  scanning:    <Loader2 size={12} className="animate-spin" />,
  analyzing:   <Loader2 size={12} className="animate-spin" />,
  scored:      <ScanSearch size={12} />,
  ai_analyzed: <Sparkles size={12} />,
  approved:    <CheckCircle size={12} />,
  error:       <AlertCircle size={12} />,
}

function imgSrc(folder_name: string, filename: string, w?: number) {
  const rel = encodeURIComponent(`${folder_name}/${filename}`)
  return `/api/media/image?rel=${rel}${w ? `&w=${w}` : ''}`
}

function scoreColor(s: number) {
  if (s >= 70) return 'text-green-600'
  if (s >= 40) return 'text-yellow-600'
  return 'text-red-500'
}

function fmtBytes(b: number) {
  if (b > 1_048_576) return `${(b / 1_048_576).toFixed(1)} MB`
  return `${(b / 1024).toFixed(0)} KB`
}

// ── Settings bar ─────────────────────────────────────────────────────────────

function SettingsBar() {
  const qc = useQueryClient()
  const { data } = useQuery<{ base_dir: string }>({
    queryKey: ['media-settings'],
    queryFn:  () => api.get('/api/media/settings'),
  })
  const [dir, setDir] = useState('')

  useEffect(() => {
    if (data?.base_dir && !dir) setDir(data.base_dir)
  }, [data])

  const save = useMutation({
    mutationFn: (base_dir: string) => api.put('/api/media/settings', { base_dir }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['media-events'] })
      qc.invalidateQueries({ queryKey: ['media-settings'] })
    },
  })

  return (
    <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-200 bg-white">
      <FolderOpen size={16} className="text-gray-400 flex-shrink-0" />
      <input
        type="text"
        placeholder="/path/to/event-folders"
        value={dir}
        onChange={e => setDir(e.target.value)}
        className="flex-1 text-sm border border-gray-200 rounded px-2 py-1 font-mono focus:outline-none focus:ring-2 focus:ring-blue-400"
      />
      <button
        onClick={() => save.mutate(dir)}
        disabled={!dir || save.isPending}
        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {save.isPending ? 'Saving…' : 'Set folder'}
      </button>
      {save.isError && (
        <span className="text-xs text-red-500">Invalid path</span>
      )}
    </div>
  )
}

// ── Event list ────────────────────────────────────────────────────────────────

function EventList({
  selected, onSelect,
}: {
  selected: string | null
  onSelect: (id: string) => void
}) {
  const { data: events = [], isLoading, refetch } = useQuery<EventSummary[]>({
    queryKey: ['media-events'],
    queryFn:  () => api.get('/api/media/events'),
    refetchInterval: 3000,
  })

  if (isLoading) return <div className="p-4 text-sm text-gray-500">Loading…</div>
  if (!events.length) return (
    <div className="p-4 text-sm text-gray-400 text-center">
      Set an image folder above to see events
    </div>
  )

  return (
    <div className="flex flex-col overflow-auto h-full">
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
          {events.length} events
        </span>
        <button onClick={() => refetch()} className="text-gray-400 hover:text-gray-600">
          <RefreshCw size={12} />
        </button>
      </div>
      {events.map(ev => (
        <button
          key={ev.event_id}
          onClick={() => onSelect(ev.event_id)}
          className={`w-full text-left px-3 py-2.5 border-b border-gray-50 hover:bg-gray-50 transition-colors ${
            selected === ev.event_id ? 'bg-blue-50 border-l-2 border-l-blue-500' : ''
          }`}
        >
          <div className="flex items-center justify-between gap-1">
            <span className="text-sm font-medium text-gray-800 truncate">{ev.event_name}</span>
            <span className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full flex-shrink-0 ${STATUS_COLOR[ev.status] ?? 'bg-gray-100 text-gray-500'}`}>
              {STATUS_ICON[ev.status]}
              {ev.status}
            </span>
          </div>
          <div className="flex items-center gap-2 mt-0.5 text-[11px] text-gray-400">
            {ev.event_date && <span>{ev.event_date}</span>}
            <span>{ev.image_count} images</span>
            {ev.hero && <span className="text-green-500">✓ hero</span>}
            {ev.gallery_count > 0 && <span>{ev.gallery_count} gallery</span>}
          </div>
        </button>
      ))}
    </div>
  )
}

// ── Thumbnail ─────────────────────────────────────────────────────────────────

function Thumb({
  folder_name, filename, w = 200, className = '', onClick, overlay,
}: {
  folder_name: string
  filename:    string
  w?:          number
  className?:  string
  onClick?:    () => void
  overlay?:    React.ReactNode
}) {
  const [err, setErr] = useState(false)
  return (
    <div className={`relative overflow-hidden bg-gray-100 ${className}`} onClick={onClick}>
      {err ? (
        <div className="flex items-center justify-center h-full">
          <ImageIcon size={20} className="text-gray-300" />
        </div>
      ) : (
        <img
          src={imgSrc(folder_name, filename, w)}
          alt={filename}
          className="w-full h-full object-cover"
          onError={() => setErr(true)}
        />
      )}
      {overlay}
    </div>
  )
}

// ── Event detail ──────────────────────────────────────────────────────────────

type Tab = 'images' | 'metadata'

function EventDetail({ event_id }: { event_id: string }) {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('images')
  const [rejectedOpen, setRejectedOpen] = useState(false)
  const [metaForm, setMetaForm] = useState<Record<string, string | string[]>>({})

  const { data: ev, isLoading } = useQuery<EventDetail>({
    queryKey: ['media-event', event_id],
    queryFn:  () => api.get(`/api/media/events/${event_id}`),
    refetchInterval: query => {
      const s = (query.state.data as EventDetail | undefined)?.status ?? ''
      return ['scanning', 'analyzing'].includes(s) ? 1000 : 5000
    },
  })

  useEffect(() => {
    if (ev?.metadata && Object.keys(metaForm).length === 0) {
      const m: Record<string, string | string[]> = {}
      for (const [k, v] of Object.entries(ev.metadata)) {
        m[k] = Array.isArray(v) ? v : String(v)
      }
      setMetaForm(m)
    }
  }, [ev?.event_id])

  const scan = useMutation({
    mutationFn: () => api.post(`/api/media/events/${event_id}/scan`, {}),
    onSuccess:  () => qc.invalidateQueries({ queryKey: ['media-event', event_id] }),
  })

  const analyze = useMutation({
    mutationFn: () => api.post(`/api/media/events/${event_id}/analyze`, {}),
    onSuccess:  () => qc.invalidateQueries({ queryKey: ['media-event', event_id] }),
  })

  const approve = useMutation({
    mutationFn: () => api.post(`/api/media/events/${event_id}/approve`, {}),
    onSuccess:  () => {
      qc.invalidateQueries({ queryKey: ['media-event', event_id] })
      qc.invalidateQueries({ queryKey: ['media-events'] })
    },
  })

  const setSelection = useMutation({
    mutationFn: (body: { hero?: string; gallery?: string[] }) =>
      api.put(`/api/media/events/${event_id}/selection`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['media-event', event_id] }),
  })

  const saveMeta = useMutation({
    mutationFn: (body: Record<string, string | string[]>) =>
      api.put(`/api/media/events/${event_id}/metadata`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['media-event', event_id] }),
  })

  if (isLoading || !ev) {
    return <div className="flex items-center justify-center h-full text-sm text-gray-400">Loading…</div>
  }

  const allImages   = Object.keys(ev.scores)
  const rejected    = ev.rejected ?? {}
  const gallery     = ev.gallery ?? []
  const hero        = ev.hero
  const rejectedList = Object.entries(rejected)
  const heroAI      = hero ? (ev.ai[hero] as AIResult | undefined) : undefined
  const ranking     = ev.ai['_ranking'] as AIRanking | undefined

  function makeHero(filename: string) {
    const newGallery = gallery.includes(filename)
      ? gallery.filter(g => g !== filename)
      : gallery
    const extendedGallery = hero ? [hero, ...newGallery].slice(0, 9) : newGallery.slice(0, 9)
    setSelection.mutate({ hero: filename, gallery: extendedGallery })
  }

  function removeFromGallery(filename: string) {
    setSelection.mutate({ gallery: gallery.filter(g => g !== filename) })
  }

  function addToGallery(filename: string) {
    if (gallery.length >= 9) return
    setSelection.mutate({ gallery: [...gallery, filename] })
  }

  const inProgress = ['scanning', 'analyzing'].includes(ev.status)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-5 pt-4 pb-3 border-b border-gray-200 bg-white flex-shrink-0">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{ev.event_name}</h2>
            <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-400">
              {ev.event_date && <span>{ev.event_date}</span>}
              <span>{ev.image_count} images</span>
              <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full font-medium ${STATUS_COLOR[ev.status] ?? ''}`}>
                {STATUS_ICON[ev.status]} {ev.status}
              </span>
            </div>
          </div>
          {/* Action bar */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => scan.mutate()}
              disabled={inProgress || scan.isPending}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 rounded disabled:opacity-40"
            >
              <ScanSearch size={13} />
              {scan.isPending || ev.status === 'scanning' ? 'Scanning…' : 'Scan'}
            </button>
            <button
              onClick={() => analyze.mutate()}
              disabled={inProgress || analyze.isPending || ev.status === 'pending'}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-indigo-100 text-indigo-700 hover:bg-indigo-200 rounded disabled:opacity-40"
            >
              <Sparkles size={13} />
              {analyze.isPending || ev.status === 'analyzing' ? 'Analyzing…' : 'AI Analysis'}
            </button>
            <button
              onClick={() => approve.mutate()}
              disabled={!hero || inProgress || approve.isPending}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs bg-green-600 text-white hover:bg-green-700 rounded disabled:opacity-40"
            >
              <Download size={13} />
              {approve.isPending ? 'Exporting…' : 'Export & Approve'}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mt-3">
          {(['images', 'metadata'] as Tab[]).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-sm pb-1 border-b-2 transition-colors ${
                tab === t ? 'border-blue-500 text-blue-600 font-medium' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {t === 'images' ? 'Images' : 'Metadata'}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto">
        {tab === 'images' && (
          <div className="p-5 space-y-6">
            {/* Hero */}
            <section>
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-1.5">
                <Star size={14} className="text-amber-500" /> Hero Image
              </h3>
              {hero ? (
                <div className="flex gap-4">
                  <div className="relative">
                    <Thumb
                      folder_name={ev.folder_name}
                      filename={hero}
                      w={480}
                      className="w-60 h-40 rounded-lg border-2 border-amber-400 cursor-pointer"
                    />
                    <span className="absolute top-1 left-1 bg-amber-400 text-xs px-1.5 py-0.5 rounded font-medium">HERO</span>
                  </div>
                  <div className="flex-1 text-xs space-y-1 text-gray-600">
                    <p className="font-medium text-gray-800 truncate">{hero}</p>
                    {ev.scores[hero] && (
                      <div className="flex gap-3 flex-wrap">
                        <span className={`font-semibold ${scoreColor(ev.scores[hero].total)}`}>
                          {ev.scores[hero].total}% overall
                        </span>
                        <span>{ev.scores[hero].width}×{ev.scores[hero].height}</span>
                        <span>{fmtBytes(ev.scores[hero].bytes)}</span>
                      </div>
                    )}
                    {heroAI && !heroAI.error && (
                      <>
                        {heroAI.hero_reason && <p className="text-gray-500 italic">{heroAI.hero_reason}</p>}
                        {heroAI.alt_text_en && <p><span className="text-gray-400">Alt EN:</span> {heroAI.alt_text_en}</p>}
                        {heroAI.alt_text_ar && <p dir="rtl"><span className="text-gray-400">Alt AR:</span> {heroAI.alt_text_ar}</p>}
                        {heroAI.caption_en && <p><span className="text-gray-400">Caption:</span> {heroAI.caption_en}</p>}
                        {heroAI.tags && heroAI.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {heroAI.tags.map((t: string) => (
                              <span key={t} className="bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded-full text-[10px]">{t}</span>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                    {ranking?.reason && (
                      <p className="text-indigo-600 text-[11px]">AI ranking: {ranking.reason}</p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-400 italic">
                  Run scan, then click an image below to set it as hero.
                </div>
              )}
            </section>

            {/* Gallery */}
            {gallery.length > 0 && (
              <section>
                <h3 className="text-sm font-semibold text-gray-700 mb-3">
                  Gallery ({gallery.length}/9)
                </h3>
                <div className="grid grid-cols-3 gap-2">
                  {gallery.map(fname => (
                    <div key={fname} className="relative group">
                      <Thumb
                        folder_name={ev.folder_name}
                        filename={fname}
                        w={300}
                        className="w-full h-28 rounded-lg cursor-pointer"
                        onClick={() => makeHero(fname)}
                        overlay={
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                            <span className="text-white text-[10px] font-medium bg-black/50 px-2 py-0.5 rounded">Make hero</span>
                          </div>
                        }
                      />
                      <button
                        onClick={() => removeFromGallery(fname)}
                        className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <X size={10} />
                      </button>
                      {ev.scores[fname] && (
                        <span className={`absolute bottom-1 left-1 text-[10px] font-bold ${scoreColor(ev.scores[fname].total)} bg-white/80 px-1 rounded`}>
                          {ev.scores[fname].total}%
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* All candidates (not hero, not in gallery, not rejected) */}
            {(() => {
              const pool = allImages.filter(
                f => f !== hero && !gallery.includes(f) && !(f in rejected)
              )
              if (!pool.length) return null
              return (
                <section>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">
                    Remaining ({pool.length})
                  </h3>
                  <div className="grid grid-cols-4 gap-2">
                    {pool.map(fname => (
                      <div key={fname} className="relative group">
                        <Thumb
                          folder_name={ev.folder_name}
                          filename={fname}
                          w={200}
                          className="w-full h-20 rounded cursor-pointer"
                          onClick={() => makeHero(fname)}
                          overlay={
                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
                          }
                        />
                        <button
                          onClick={e => { e.stopPropagation(); addToGallery(fname) }}
                          disabled={gallery.length >= 9}
                          className="absolute top-1 right-1 bg-blue-500 text-white rounded text-[9px] px-1 opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                        >
                          +gallery
                        </button>
                        {ev.scores[fname] && (
                          <span className={`absolute bottom-1 left-1 text-[10px] font-bold ${scoreColor(ev.scores[fname].total)} bg-white/80 px-1 rounded`}>
                            {ev.scores[fname].total}%
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )
            })()}

            {/* Rejected */}
            {rejectedList.length > 0 && (
              <section>
                <button
                  onClick={() => setRejectedOpen(v => !v)}
                  className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700"
                >
                  {rejectedOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  Rejected ({rejectedList.length})
                </button>
                {rejectedOpen && (
                  <div className="mt-2 grid grid-cols-4 gap-2">
                    {rejectedList.map(([fname, reason]) => (
                      <div key={fname} className="relative group opacity-50 hover:opacity-90">
                        <Thumb
                          folder_name={ev.folder_name}
                          filename={fname}
                          w={200}
                          className="w-full h-20 rounded grayscale"
                        />
                        <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[9px] px-1 py-0.5 truncate">
                          {reason}
                        </div>
                        <button
                          onClick={() => addToGallery(fname)}
                          disabled={gallery.length >= 9}
                          className="absolute top-1 right-1 bg-blue-500 text-white rounded text-[9px] px-1 opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                        >
                          restore
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            )}
          </div>
        )}

        {tab === 'metadata' && (
          <div className="p-5 max-w-xl space-y-4">
            {[
              { key: 'eventSlug', label: 'Event Slug', placeholder: 'graduation-ceremony-2026' },
              { key: 'eventDate', label: 'Event Date', placeholder: '2026-06-15' },
              { key: 'eventName', label: 'Event Name', placeholder: ev.event_name },
              { key: 'altTextEn', label: 'Alt Text (EN)', placeholder: '' },
              { key: 'altTextAr', label: 'Alt Text (AR)', placeholder: '', rtl: true },
              { key: 'captionEn', label: 'Caption (EN)', placeholder: '' },
              { key: 'captionAr', label: 'Caption (AR)', placeholder: '', rtl: true },
            ].map(({ key, label, placeholder, rtl }) => (
              <div key={key}>
                <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
                <input
                  type="text"
                  value={typeof metaForm[key] === 'string' ? metaForm[key] as string : ''}
                  onChange={e => setMetaForm(prev => ({ ...prev, [key]: e.target.value }))}
                  placeholder={placeholder}
                  dir={rtl ? 'rtl' : undefined}
                  className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
                />
              </div>
            ))}

            {/* Tags */}
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Tags (comma-separated)</label>
              <input
                type="text"
                value={Array.isArray(metaForm.tags) ? (metaForm.tags as string[]).join(', ') : ''}
                onChange={e =>
                  setMetaForm(prev => ({
                    ...prev,
                    tags: e.target.value.split(',').map(t => t.trim()).filter(Boolean),
                  }))
                }
                className="w-full border border-gray-200 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>

            <button
              onClick={() => saveMeta.mutate(metaForm)}
              disabled={saveMeta.isPending}
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {saveMeta.isPending ? 'Saving…' : 'Save Metadata'}
            </button>
            {saveMeta.isSuccess && (
              <span className="text-xs text-green-600 ml-2">Saved</span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function MediaLibrary() {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  return (
    <div className="flex flex-col h-full">
      <SettingsBar />
      <div className="flex flex-1 overflow-hidden">
        {/* Left: event list */}
        <div className="w-72 flex-shrink-0 border-r border-gray-200 overflow-auto bg-white">
          <EventList selected={selectedId} onSelect={setSelectedId} />
        </div>

        {/* Right: event detail */}
        <div className="flex-1 overflow-hidden bg-gray-50">
          {selectedId ? (
            <EventDetail event_id={selectedId} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-sm text-gray-400 gap-2">
              <ImageIcon size={32} className="opacity-30" />
              Select an event to view and curate images
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
