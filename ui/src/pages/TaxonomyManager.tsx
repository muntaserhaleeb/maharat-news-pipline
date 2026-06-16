import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronDown, ChevronUp, ChevronRight, Pencil, Trash2, Plus, X, Check, ArrowUp, ArrowDown } from 'lucide-react'
import { api } from '../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface CategoryRule {
  name:       string
  keywords:   string[]
  post_count: number
}

interface CategoriesData {
  default:       string
  default_count: number
  ordered_rules: CategoryRule[]
}

interface TaxonomyResponse {
  categories:        CategoriesData
  tags:              Record<string, string[]>
  total_categories:  number
  total_tags:        number
}

const TABS = ['Categories', 'Tags'] as const
type Tab = typeof TABS[number]

// ── Category edit form ────────────────────────────────────────────────────────

function CategoryForm({
  initial, onSave, onCancel,
}: {
  initial: { name: string; keywords: string[] }
  onSave:  (name: string, keywords: string[]) => void
  onCancel: () => void
}) {
  const [name, setName] = useState(initial.name)
  const [kwText, setKwText] = useState(initial.keywords.join('\n'))

  const save = () => {
    const keywords = kwText
      .split('\n')
      .map(k => k.trim())
      .filter(Boolean)
    onSave(name.trim(), keywords)
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 space-y-3">
      <div>
        <label className="block text-[11px] font-medium text-gray-600 mb-1">Category name *</label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. Events & Ceremonies"
        />
      </div>
      <div>
        <label className="block text-[11px] font-medium text-gray-600 mb-1">
          Keywords — one per line, case-insensitive substring match
        </label>
        <textarea
          value={kwText}
          onChange={e => setKwText(e.target.value)}
          rows={8}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
          placeholder="graduation ceremony&#10;graduation of its first&#10;celebrates intake"
        />
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!name.trim()}
          onClick={save}
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

// ── Category row ──────────────────────────────────────────────────────────────

function CategoryRow({
  rule, index, total,
  onEdit, onDelete, onMove,
}: {
  rule:    CategoryRule
  index:   number
  total:   number
  onEdit:  () => void
  onDelete: () => void
  onMove:  (dir: -1 | 1) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3 bg-white hover:bg-gray-50 group">
        {/* index badge */}
        <span className="text-[11px] font-mono text-gray-400 tabular-nums w-5 text-right flex-shrink-0">
          {index + 1}
        </span>

        {/* expand toggle */}
        <button
          onClick={() => setExpanded(e => !e)}
          className="flex-1 flex items-center gap-2 text-left min-w-0"
        >
          {expanded
            ? <ChevronDown size={13} className="text-gray-400 flex-shrink-0" />
            : <ChevronRight size={13} className="text-gray-400 flex-shrink-0" />
          }
          <span className="text-sm font-semibold text-gray-900 truncate">{rule.name}</span>
        </button>

        {/* stats */}
        <div className="flex items-center gap-3 flex-shrink-0 text-xs text-gray-400">
          <span>{rule.keywords.length} kw</span>
          <span className="bg-blue-50 text-blue-600 px-2 py-0.5 rounded-full tabular-nums">
            {rule.post_count} posts
          </span>
        </div>

        {/* actions (show on hover) */}
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onMove(-1)}
            disabled={index === 0}
            className="p-1 rounded text-gray-400 hover:text-gray-700 hover:bg-gray-100 disabled:opacity-20"
            title="Move up"
          >
            <ArrowUp size={12} />
          </button>
          <button
            onClick={() => onMove(1)}
            disabled={index === total - 1}
            className="p-1 rounded text-gray-400 hover:text-gray-700 hover:bg-gray-100 disabled:opacity-20"
            title="Move down"
          >
            <ArrowDown size={12} />
          </button>
          <button
            onClick={onEdit}
            className="p-1 rounded text-gray-400 hover:text-blue-600 hover:bg-blue-50"
            title="Edit"
          >
            <Pencil size={12} />
          </button>
          <button
            onClick={onDelete}
            className="p-1 rounded text-gray-400 hover:text-red-600 hover:bg-red-50"
            title="Delete"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {expanded && (
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
          <div className="flex flex-wrap gap-1.5">
            {rule.keywords.map(kw => (
              <span key={kw} className="text-xs bg-white border border-gray-200 text-gray-600 px-2 py-0.5 rounded font-mono">
                {kw}
              </span>
            ))}
            {rule.keywords.length === 0 && (
              <span className="text-xs text-gray-400 italic">No keywords defined</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Tag group card ────────────────────────────────────────────────────────────

function TagGroup({
  group, tags, onAdd, onRemove,
}: {
  group:    string
  tags:     string[]
  onAdd:    (tag: string) => void
  onRemove: (tag: string) => void
}) {
  const [input, setInput] = useState('')

  const submit = () => {
    const v = input.trim()
    if (v) { onAdd(v); setInput('') }
  }

  const label = group.replace(/_/g, ' ')

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-800 capitalize">{label}</h3>
        <span className="text-[11px] text-gray-400 tabular-nums">{tags.length}</span>
      </div>

      <div className="flex flex-wrap gap-1.5 mb-3 min-h-[24px]">
        {tags.map(tag => (
          <span
            key={tag}
            className="inline-flex items-center gap-1 bg-gray-100 text-gray-700 text-xs px-2 py-0.5 rounded-full"
          >
            {tag}
            <button
              type="button"
              onClick={() => onRemove(tag)}
              className="hover:text-red-500 transition-colors leading-none"
            >
              <X size={10} />
            </button>
          </span>
        ))}
        {tags.length === 0 && (
          <span className="text-xs text-gray-300 italic">empty</span>
        )}
      </div>

      <div className="flex gap-1.5">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); submit() } }}
          placeholder="New tag…"
          className="flex-1 border border-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <button
          type="button"
          onClick={submit}
          className="p-1.5 bg-gray-100 hover:bg-gray-200 rounded text-gray-600"
          title="Add"
        >
          <Plus size={12} />
        </button>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function TaxonomyManager() {
  const qc = useQueryClient()
  const [activeTab,    setActiveTab]    = useState<Tab>('Categories')
  const [editingIdx,   setEditingIdx]   = useState<number | null>(null)
  const [addingNew,    setAddingNew]    = useState(false)
  // local copy of rules for reordering without re-fetching each time
  const [localRules,   setLocalRules]   = useState<CategoryRule[] | null>(null)

  const { data, isLoading } = useQuery<TaxonomyResponse>({
    queryKey: ['taxonomy'],
    queryFn:  () => api.get<TaxonomyResponse>('/taxonomy'),
  })

  useEffect(() => {
    if (data && !localRules) {
      setLocalRules(data.categories.ordered_rules)
    }
  }, [data])

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['taxonomy'] })
    setLocalRules(null)
  }

  const rules      = localRules ?? data?.categories?.ordered_rules ?? []
  const defaultCat = data?.categories?.default ?? 'General'
  const tags       = data?.tags ?? {}

  // ── category mutations ──
  const saveCatsMut = useMutation({
    mutationFn: (newRules: CategoryRule[]) =>
      api.put('/taxonomy/categories', {
        default: defaultCat,
        ordered_rules: newRules.map(r => ({ name: r.name, keywords: r.keywords })),
      }),
    onSuccess: invalidate,
  })

  const saveRules = (newRules: CategoryRule[]) => {
    setLocalRules(newRules)
    saveCatsMut.mutate(newRules)
  }

  const handleMove = (index: number, dir: -1 | 1) => {
    const next = [...rules]
    const swap = index + dir
    if (swap < 0 || swap >= next.length) return
    ;[next[index], next[swap]] = [next[swap], next[index]]
    saveRules(next)
  }

  const handleSaveCategory = (index: number, name: string, keywords: string[]) => {
    const next = rules.map((r, i) =>
      i === index ? { ...r, name, keywords } : r
    )
    saveRules(next)
    setEditingIdx(null)
  }

  const handleAddCategory = (name: string, keywords: string[]) => {
    const newRule: CategoryRule = { name, keywords, post_count: 0 }
    saveRules([...rules, newRule])
    setAddingNew(false)
  }

  const handleDeleteCategory = (index: number) => {
    if (!confirm(`Delete category "${rules[index].name}"?`)) return
    saveRules(rules.filter((_, i) => i !== index))
  }

  // ── tag mutations ──
  const addTagMut = useMutation({
    mutationFn: ({ group, tag }: { group: string; tag: string }) =>
      api.post(`/taxonomy/tags/${encodeURIComponent(group)}`, { tag }),
    onSuccess: invalidate,
  })

  const removeTagMut = useMutation({
    mutationFn: ({ group, tag }: { group: string; tag: string }) =>
      api.post(`/taxonomy/tags/${encodeURIComponent(group)}/remove`, { tag }),
    onSuccess: invalidate,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Loading taxonomy…
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4 border-b border-gray-200 bg-white flex-shrink-0">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Taxonomy Manager</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Edit categories and tags in <code className="font-mono">config/taxonomy.yaml</code>
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span className="bg-gray-100 px-2 py-1 rounded-lg">
            {data?.total_categories ?? 0} categories
          </span>
          <span className="bg-gray-100 px-2 py-1 rounded-lg">
            {data?.total_tags ?? 0} tags
          </span>
        </div>
      </div>

      {/* tabs */}
      <div className="border-b border-gray-200 bg-white px-6 flex-shrink-0">
        <div className="flex gap-0">
          {TABS.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* scrollable content */}
      <div className="flex-1 overflow-auto px-6 py-5">
        {/* ── Categories tab ── */}
        {activeTab === 'Categories' && (
          <div className="max-w-3xl space-y-2">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs text-gray-500">
                Rules are evaluated in order — first match wins. Default category:{' '}
                <strong>{defaultCat}</strong> ({data?.categories?.default_count ?? 0} posts)
              </p>
              <button
                onClick={() => setAddingNew(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-lg"
              >
                <Plus size={12} /> Add Category
              </button>
            </div>

            {rules.map((rule, idx) =>
              editingIdx === idx ? (
                <CategoryForm
                  key={idx}
                  initial={{ name: rule.name, keywords: rule.keywords }}
                  onSave={(name, kws) => handleSaveCategory(idx, name, kws)}
                  onCancel={() => setEditingIdx(null)}
                />
              ) : (
                <CategoryRow
                  key={idx}
                  rule={rule}
                  index={idx}
                  total={rules.length}
                  onEdit={() => setEditingIdx(idx)}
                  onDelete={() => handleDeleteCategory(idx)}
                  onMove={dir => handleMove(idx, dir)}
                />
              )
            )}

            {addingNew && (
              <CategoryForm
                initial={{ name: '', keywords: [] }}
                onSave={handleAddCategory}
                onCancel={() => setAddingNew(false)}
              />
            )}

            {saveCatsMut.error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700">
                {saveCatsMut.error.message}
              </div>
            )}
          </div>
        )}

        {/* ── Tags tab ── */}
        {activeTab === 'Tags' && (
          <div>
            <p className="text-xs text-gray-500 mb-4">
              Tags are vocabulary lists organised by group. Press Enter or click + to add a tag. Click × to remove.
            </p>
            <div className="grid grid-cols-2 gap-4 xl:grid-cols-3">
              {Object.entries(tags).map(([group, groupTags]) => (
                <TagGroup
                  key={group}
                  group={group}
                  tags={groupTags}
                  onAdd={tag => addTagMut.mutate({ group, tag })}
                  onRemove={tag => removeTagMut.mutate({ group, tag })}
                />
              ))}
            </div>
            {(addTagMut.error || removeTagMut.error) && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-3 text-sm text-red-700 mt-4">
                {(addTagMut.error ?? removeTagMut.error)?.message}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
