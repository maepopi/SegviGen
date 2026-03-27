import { useState, useCallback } from 'react'
import { uploadFile } from './api/client'
import { FullTab }        from './components/tabs/FullTab'
import { Full2DTab }      from './components/tabs/Full2DTab'
import { Viewer3D }       from './components/Viewer3D'
import { Upload, Grid2x2, Wand2 } from 'lucide-react'

type TabId = 'full' | 'full2d'

const TABS: Array<{ id: TabId; label: string; icon: React.ReactNode }> = [
  { id: 'full',        label: 'Full',              icon: <Grid2x2 size={15} /> },
  { id: 'full2d',      label: 'Full + 2D Map',     icon: <Wand2 size={15} /> },
]

export default function App() {
  const [activeTab,    setActiveTab]    = useState<TabId>('full')
  const [uploadedPath, setUploadedPath] = useState<string | null>(null)
  const [uploadedName, setUploadedName] = useState<string | null>(null)
  const [dragOver,     setDragOver]     = useState(false)

  const handleUpload = useCallback(async (file: File) => {
    setUploadedName('Uploading…')
    const path = await uploadFile(file)
    setUploadedPath(path)
    setUploadedName(file.name)
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [handleUpload])

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
  }, [handleUpload])

  return (
    <div className="flex flex-col h-full bg-bg">

      {/* ── Header ── */}
      <header className="flex items-center gap-3 px-5 border-b border-border bg-bg/90 backdrop-blur-sm sticky top-0 z-50 shrink-0"
        style={{ height: '52px' }}>
        <span className="text-accent font-bold text-lg tracking-tight">SegviGen</span>
        <span className="text-dim text-xs">3D Part Segmentation</span>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ── Sidebar ── */}
        <aside className="w-64 shrink-0 bg-card border-r border-border flex flex-col gap-4 p-3 overflow-y-auto">

          {/* Upload */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-dim mb-2">Input Model</p>
            <label
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              className={`flex items-center gap-3 border-2 border-dashed rounded-xl px-3 py-2 cursor-pointer transition-all
                ${dragOver ? 'border-accent bg-accent/10' : 'border-border hover:border-accent/50 hover:bg-hover'}`}
            >
              <input type="file" accept=".glb,.obj,.ply" className="hidden" onChange={onFileChange} />
              <Upload size={16} className="text-muted shrink-0" />
              <p className="text-xs text-muted leading-tight truncate">
                {uploadedName ?? 'Drop GLB / OBJ / PLY'}
              </p>
            </label>

            <div className="mt-2 rounded-xl overflow-hidden border border-border" style={{ height: '200px' }}>
              <Viewer3D filePath={uploadedPath} compact />
            </div>
          </div>

          {/* Nav */}
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-dim mb-2">Mode</p>
            <nav className="flex flex-col gap-1">
              {TABS.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-left transition-all
                    ${activeTab === tab.id
                      ? 'bg-accent/10 text-accent font-semibold'
                      : 'text-muted hover:bg-hover hover:text-white'}`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          <div className="mt-auto text-[11px] text-dim leading-relaxed">
            Base models cached on first run.<br />
            Checkpoints cached per path.
          </div>
        </aside>

        {/* ── Main content ── */}
        <main className="flex-1 overflow-y-auto p-6">
          {activeTab === 'full'        && <FullTab        glbPath={uploadedPath} />}
          {activeTab === 'full2d'      && <Full2DTab      glbPath={uploadedPath} />}
        </main>
      </div>
    </div>
  )
}
