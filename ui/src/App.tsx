import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ConfigManager from './pages/ConfigManager'
import RetrievalPlayground from './pages/RetrievalPlayground'
import Generator from './pages/Generator'
import EntityManager from './pages/EntityManager'
import TaxonomyManager from './pages/TaxonomyManager'
import PipelineRunner from './pages/PipelineRunner'
import ReviewQueue from './pages/ReviewQueue'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/"          element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/config"    element={<ConfigManager />} />
          <Route path="/retrieval" element={<RetrievalPlayground />} />
          <Route path="/generator" element={<Generator />} />
          <Route path="/entities"  element={<EntityManager />} />
          <Route path="/taxonomy"  element={<TaxonomyManager />} />
          <Route path="/pipeline"  element={<PipelineRunner />} />
          <Route path="/review"    element={<ReviewQueue />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
