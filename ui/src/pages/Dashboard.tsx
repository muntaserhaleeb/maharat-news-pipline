import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { CheckCircle2, XCircle, FileCode2, FolderOpen, Image, BookOpen, Settings2, SearchCode, Wand2, ArrowRight } from 'lucide-react'
import { api } from '../api/client'

interface ConfigFile {
  name: string
  exists: boolean
  size_kb: number
}

interface StatusResponse {
  phase: string
  data: { posts: number; images: number; knowledge_docs: number }
  configs: {
    managed: number
    present: number
    missing: string[]
    files: ConfigFile[]
  }
}

function StatusDot({ ok }: { ok: boolean }) {
  return ok
    ? <CheckCircle2 size={15} className="text-green-500 flex-shrink-0" />
    : <XCircle     size={15} className="text-red-400  flex-shrink-0" />
}

export default function Dashboard() {
  const health = useQuery<{ status: string }>({
    queryKey: ['health'],
    queryFn:  () => api.get<{ status: string }>('/health'),
    refetchInterval: 30_000,
  })

  const status = useQuery<StatusResponse>({
    queryKey: ['dashboard-status'],
    queryFn:  () => api.get<StatusResponse>('/dashboard/status'),
  })

  const apiOk = health.data?.status === 'ok'

  return (
    <div className="p-8 max-w-4xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-400 mt-1">Phase 1 — config management foundation</p>
      </div>

      {/* System status */}
      <section>
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">System</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 flex items-center gap-3">
            <StatusDot ok={apiOk} />
            <div>
              <p className="text-sm font-semibold text-gray-800">Backend API</p>
              <p className="text-xs text-gray-400">{health.isLoading ? 'Checking…' : apiOk ? 'Running on :8000' : 'Unreachable'}</p>
            </div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 flex items-center gap-3">
            <StatusDot ok={!status.isError} />
            <div>
              <p className="text-sm font-semibold text-gray-800">Database</p>
              <p className="text-xs text-gray-400">SQLite · storage/maharat_ops.db</p>
            </div>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 px-4 py-3 flex items-center gap-3">
            <StatusDot ok={(status.data?.configs.missing.length ?? 1) === 0} />
            <div>
              <p className="text-sm font-semibold text-gray-800">Config Files</p>
              <p className="text-xs text-gray-400">
                {status.isLoading
                  ? 'Loading…'
                  : `${status.data?.configs.present ?? 0} / ${status.data?.configs.managed ?? 0} present`}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Data overview */}
      {status.data && (
        <section>
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Data</h2>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Posts',          value: status.data.data.posts,          Icon: FolderOpen },
              { label: 'Images',         value: status.data.data.images,         Icon: Image },
              { label: 'Knowledge Docs', value: status.data.data.knowledge_docs, Icon: BookOpen },
            ].map(({ label, value, Icon }) => (
              <div key={label} className="bg-white rounded-xl border border-gray-200 px-5 py-4">
                <Icon size={16} className="text-gray-300 mb-2" />
                <p className="text-2xl font-bold text-gray-900 tabular-nums">{value.toLocaleString()}</p>
                <p className="text-xs text-gray-400 mt-0.5">{label}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Config files */}
      {status.data && (
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Config Files</h2>
            <Link to="/config" className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 font-medium">
              Manage <ArrowRight size={11} />
            </Link>
          </div>
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-100">
                <tr>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">File</th>
                  <th className="text-right px-4 py-2.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">Size</th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-gray-400 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {status.data.configs.files.map((f) => (
                  <tr key={f.name} className="hover:bg-gray-50/50">
                    <td className="px-4 py-2.5 flex items-center gap-2">
                      <FileCode2 size={13} className="text-gray-300 flex-shrink-0" />
                      <span className="font-mono text-xs text-gray-700">{f.name}</span>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-400 tabular-nums">
                      {f.exists ? `${f.size_kb} KB` : '—'}
                    </td>
                    <td className="px-4 py-2.5">
                      {f.exists
                        ? <span className="text-xs text-green-600 font-medium">present</span>
                        : <span className="text-xs text-red-400 font-medium">missing</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Coming next */}
      <section>
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Coming in Phase 2</h2>
        <div className="grid grid-cols-2 gap-3">
          {[
            { Icon: SearchCode, label: 'Retrieval Playground', desc: 'Test hybrid search against Qdrant' },
            { Icon: Wand2,      label: 'Content Generator',    desc: 'Draft articles from retrieved context' },
          ].map(({ Icon, label, desc }) => (
            <div key={label} className="bg-white rounded-xl border border-dashed border-gray-200 px-5 py-4 opacity-50">
              <Icon size={16} className="text-gray-300 mb-2" />
              <p className="text-sm font-semibold text-gray-500">{label}</p>
              <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
