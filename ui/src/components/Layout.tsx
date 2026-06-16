import { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Settings2,
  SearchCode,
  Wand2,
  Users,
  Tags,
  Terminal,
  Inbox,
} from 'lucide-react'

interface Props { children: ReactNode }

const ACTIVE_NAV = [
  { to: '/dashboard', label: 'Dashboard',      Icon: LayoutDashboard },
  { to: '/config',    label: 'Config Manager', Icon: Settings2 },
  { to: '/retrieval', label: 'Retrieval',      Icon: SearchCode },
  { to: '/generator', label: 'Generator',      Icon: Wand2 },
  { to: '/entities',  label: 'Entities',       Icon: Users },
  { to: '/taxonomy',  label: 'Taxonomy',        Icon: Tags },
  { to: '/pipeline',  label: 'Pipeline',        Icon: Terminal },
  { to: '/review',    label: 'Review Queue',    Icon: Inbox },
]

export default function Layout({ children }: Props) {
  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <aside className="w-52 bg-slate-900 text-white flex flex-col flex-shrink-0">
        <div className="px-4 py-4 border-b border-slate-700/60">
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Maharat</p>
          <p className="text-base font-bold text-white mt-0.5 leading-tight">RAG Ops</p>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-auto">
          {ACTIVE_NAV.map(({ to, label, Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                }`
              }
            >
              <Icon size={14} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="px-4 py-3 border-t border-slate-700/60 text-[11px] text-slate-600">
          Internal tool · v1.0
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  )
}
