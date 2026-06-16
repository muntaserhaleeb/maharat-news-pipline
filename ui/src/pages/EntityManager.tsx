import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Pencil, Trash2, X, Check, AlertTriangle, Building2, BookOpen, MapPin, Award, User } from 'lucide-react'
import { api } from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface EntityItem {
  canonical:   string
  aliases:     string[]
  title?:      string | null
  affiliation?: string | null
}

interface EntityMap {
  organizations: EntityItem[]
  programs:      EntityItem[]
  locations:     EntityItem[]
  credentials:   EntityItem[]
  people:        EntityItem[]
}

interface Duplicate {
  alias:    string
  entity_1: { type: string; canonical: string }
  entity_2: { type: string; canonical: string }
}

interface EntitiesResponse {
  entities:     EntityMap
  duplicates:   Duplicate[]
  entity_types: string[]
}

type EntityType = keyof EntityMap

const TYPE_LABELS: Record<EntityType, string> = {
  organizations: 'Organizations',
  programs:      'Programs',
  locations:     'Locations',
  credentials:   'Credentials',
  people:        'People',
}

const TYPE_ICONS: Record<EntityType, React.ElementType> = {
  organizations: Building2,
  programs:      BookOpen,
  locations:     MapPin,
  credentials:   Award,
  people:        User,
}

const TYPE_COLOURS: Record<EntityType, string> = {
  organizations: 'bg-purple-50 text-purple-700 border-purple-200',
  programs:      'bg-teal-50 text-teal-700 border-teal-200',
  locations:     'bg-orange-50 text-orange-700 border-orange-200',
  credentials:   'bg-sky-50 text-sky-700 border-sky-200',
  people:        'bg-pink-50 text-pink-700 border-pink-200',
}

const EMPTY_FORM: EntityItem = { canonical: '', aliases: [], title: '', affiliation: '' }

// ── Alias chip input ──────────────────────────────────────────────────────────

function AliasInput({
  aliases, onChange,
}: { aliases: string[]; onChange: (a: string[]) => void }) {
  const [input, setInput] = useState('')

  const add = () => {
    const v = input.trim()
    if (v && !aliases.includes(v)) onChange([...aliases, v])
    setInput('')
  }

  return (
    <div>
      <div className="flex flex-wrap gap-1 mb-1.5 min-h-[24px]">
        {aliases.map(a => (
          <span key={a} className="inline-flex items-center gap-1 bg-gray-100 text-gray-700 text-xs px-2 py-0.5 rounded-full">
            {a}
            <button type="button" onClick={() => onChange(aliases.filter(x => x !== a))} className="hover:text-red-500 leading-none">
              <X size={10} />
            </button>
          </span>
        ))}
        {aliases.length === 0 && <span className="text-xs text-gray-300 italic">no aliases</span>}
      </div>
      <div className="flex gap-1.5">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); add() } }}
          placeholder="Add alias…"
          className="flex-1 border border-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="button"
          onClick={add}
          className="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-xs text-gray-600"
        >
          Add
        </button>
      </div>
    </div>
  )
}

// ── Inline entity form (add or edit) ─────────────────────────────────────────

function EntityForm({
  initial, onSave, onCancel, isPeople,
}: {
  initial: EntityItem
  onSave: (item: EntityItem) => void
  onCancel: () => void
  isPeople: boolean
}) {
  const [form, setForm] = useState<EntityItem>(initial)

  const set = (k: keyof EntityItem, v: string | string[]) =>
    setForm(f => ({ ...f, [k]: v }))

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-[11px] font-medium text-gray-600 mb-1">Canonical name *</label>
          <input
            value={form.canonical}
            onChange={e => set('canonical', e.target.value)}
            placeholder="Full authoritative name"
            className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        {isPeople && (
          <div>
            <label className="block text-[11px] font-medium text-gray-600 mb-1">Title</label>
            <input
              value={form.title || ''}
              onChange={e => set('title', e.target.value)}
              placeholder="Role / title"
              className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        )}
      </div>

      {isPeople && (
        <div>
          <label className="block text-[11px] font-medium text-gray-600 mb-1">Affiliation</label>
          <input
            value={form.affiliation || ''}
            onChange={e => set('affiliation', e.target.value)}
            placeholder="Organization and department"
            className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      <div>
        <label className="block text-[11px] font-medium text-gray-600 mb-1">Aliases</label>
        <AliasInput aliases={form.aliases} onChange={v => set('aliases', v)} />
      </div>

      <div className="flex gap-2 pt-1">
        <button
          type="button"
          disabled={!form.canonical.trim()}
          onClick={() => onSave(form)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-lg disabled:opacity-40 transition-colors"
        >
          <Check size={12} /> Save
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-3 py-1.5 text-xs text-gray-600 hover:text-gray-800 border border-gray-300 rounded-lg"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

// ── Entity row ────────────────────────────────────────────────────────────────

function EntityRow({
  item, index, type, counts, onEdit, onDelete,
}: {
  item:     EntityItem
  index:    number
  type:     EntityType
  counts:   Record<string, number>
  onEdit:   () => void
  onDelete: () => void
}) {
  const mentionKey = item.canonical.toLowerCase()
  const count = counts[mentionKey]
  const colourClass = TYPE_COLOURS[type]

  return (
    <div className="flex items-start gap-3 py-3 px-4 hover:bg-gray-50 rounded-xl group">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold text-gray-900">{item.canonical}</span>
          {item.title && (
            <span className="text-xs text-gray-400 italic">{item.title}</span>
          )}
        </div>
        {item.affiliation && (
          <p className="text-xs text-gray-400 mt-0.5">{item.affiliation}</p>
        )}
        {item.aliases.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1.5">
            {item.aliases.map(a => (
              <span key={a} className={`text-[11px] px-1.5 py-0.5 rounded-full border ${colourClass}`}>
                {a}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 flex-shrink-0 mt-0.5">
        {count !== undefined ? (
          <span className="text-xs tabular-nums text-gray-400" title={`${count} articles mention this entity`}>
            {count} {count === 1 ? 'article' : 'articles'}
          </span>
        ) : (
          <span className="text-xs text-gray-300">—</span>
        )}
        <button
          onClick={onEdit}
          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-gray-400 hover:text-blue-600 hover:bg-blue-50 transition-all"
          title="Edit"
        >
          <Pencil size={13} />
        </button>
        <button
          onClick={onDelete}
          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-all"
          title="Delete"
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function EntityManager() {
  const qc = useQueryClient()

  const [activeType, setActiveType]   = useState<EntityType>('organizations')
  const [editingIdx, setEditingIdx]   = useState<number | null>(null) // null=none, -1=adding new
  const [showDupes,  setShowDupes]    = useState(true)

  const { data, isLoading } = useQuery<EntitiesResponse>({
    queryKey: ['entities'],
    queryFn:  () => api.get<EntitiesResponse>('/entities'),
  })

  const { data: counts = {} } = useQuery<Record<string, number>>({
    queryKey: ['entity-counts'],
    queryFn:  () => api.get<Record<string, number>>('/entities/counts'),
    staleTime: 5 * 60 * 1000,
  })

  const invalidate = () => qc.invalidateQueries({ queryKey: ['entities'] })

  const addMut = useMutation({
    mutationFn: (item: EntityItem) =>
      api.post(`/entities/${activeType}`, { entity: item }),
    onSuccess: () => { invalidate(); setEditingIdx(null) },
  })

  const updateMut = useMutation({
    mutationFn: ({ index, item }: { index: number; item: EntityItem }) =>
      api.put(`/entities/${activeType}/${index}`, { entity: item }),
    onSuccess: () => { invalidate(); setEditingIdx(null) },
  })

  const deleteMut = useMutation({
    mutationFn: (index: number) =>
      api.delete<unknown>(`/entities/${activeType}/${index}`),
    onSuccess: invalidate,
  })

  const items      = data?.entities[activeType] ?? []
  const duplicates = data?.duplicates ?? []

  const isPeople = activeType === 'people'

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Loading entities…
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── left: type sidebar ── */}
      <aside className="w-48 flex-shrink-0 bg-white border-r border-gray-200 pt-5 px-2">
        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest px-3 mb-2">
          Entity Types
        </p>
        {(Object.keys(TYPE_LABELS) as EntityType[]).map(type => {
          const count = data?.entities[type]?.length ?? 0
          const Icon  = TYPE_ICONS[type]
          const active = type === activeType
          return (
            <button
              key={type}
              onClick={() => { setActiveType(type); setEditingIdx(null) }}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium mb-0.5 transition-colors ${
                active
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Icon size={13} />
              <span className="flex-1 text-left">{TYPE_LABELS[type]}</span>
              <span className={`text-[11px] tabular-nums ${active ? 'text-blue-200' : 'text-gray-400'}`}>
                {count}
              </span>
            </button>
          )
        })}
      </aside>

      {/* ── right: entity list ── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-gray-200 bg-white flex-shrink-0">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Entity Manager</h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Manage canonical names and aliases in <code className="font-mono">config/entities.yaml</code>
            </p>
          </div>
          <button
            onClick={() => setEditingIdx(editingIdx === -1 ? null : -1)}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <Plus size={14} />
            Add {TYPE_LABELS[activeType].replace(/s$/, '')}
          </button>
        </div>

        {/* scrollable body */}
        <div className="flex-1 overflow-auto px-6 py-4 space-y-3">
          {/* duplicate warnings */}
          {showDupes && duplicates.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={15} className="text-yellow-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm font-medium text-yellow-800">
                    {duplicates.length} duplicate alias{duplicates.length > 1 ? 'es' : ''} detected
                  </p>
                </div>
                <button onClick={() => setShowDupes(false)} className="text-yellow-500 hover:text-yellow-700">
                  <X size={14} />
                </button>
              </div>
              <ul className="mt-2 space-y-1">
                {duplicates.map((d, i) => (
                  <li key={i} className="text-xs text-yellow-700">
                    <span className="font-semibold">"{d.alias}"</span> appears in{' '}
                    <span className="italic">{d.entity_1.type}/{d.entity_1.canonical}</span> and{' '}
                    <span className="italic">{d.entity_2.type}/{d.entity_2.canonical}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* add-new form */}
          {editingIdx === -1 && (
            <EntityForm
              initial={EMPTY_FORM}
              isPeople={isPeople}
              onSave={item => addMut.mutate(item)}
              onCancel={() => setEditingIdx(null)}
            />
          )}

          {/* entity list */}
          {items.length === 0 && editingIdx !== -1 ? (
            <p className="text-sm text-gray-400 text-center py-12">
              No {TYPE_LABELS[activeType].toLowerCase()} yet. Click "Add" to create one.
            </p>
          ) : (
            <div className="bg-white border border-gray-200 rounded-xl divide-y divide-gray-100">
              {items.map((item, idx) =>
                editingIdx === idx ? (
                  <div key={idx} className="p-4">
                    <EntityForm
                      initial={item}
                      isPeople={isPeople}
                      onSave={updated => updateMut.mutate({ index: idx, item: updated })}
                      onCancel={() => setEditingIdx(null)}
                    />
                  </div>
                ) : (
                  <EntityRow
                    key={idx}
                    item={item}
                    index={idx}
                    type={activeType}
                    counts={counts}
                    onEdit={() => setEditingIdx(idx)}
                    onDelete={() => {
                      if (confirm(`Delete "${item.canonical}"?`)) deleteMut.mutate(idx)
                    }}
                  />
                )
              )}
            </div>
          )}

          {/* mutation error */}
          {(addMut.error || updateMut.error || deleteMut.error) && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700">
              {(addMut.error ?? updateMut.error ?? deleteMut.error)?.message}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
