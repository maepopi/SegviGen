/**
 * Images → Parts JSON.
 * A model-free mode: upload one or more images and run the VLM analysis to get
 * back the assembly-tree JSON describing each part (name, color, material).
 * No 3D model and no guidance-map image — analysis only.
 */
import { useRef, useState } from 'react'
import { Field, TextInput, Select, SliderField, Btn, StatusBadge } from '../ui/Field'
import { useJob } from '../../hooks/useJob'
import { fileUrl, uploadFile } from '../../api/client'
import { Download } from 'lucide-react'

interface PartsResult { description: unknown }

function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function ImagesTab() {
  const [imgs,         setImgs]         = useState<string[]>([])
  const [newImg,       setNewImg]       = useState('')
  const [apiKey,       setApiKey]       = useState('')
  const [gridCols,     setGridCols]     = useState(2)
  const [analyzeModel, setAnalyzeModel] = useState('gemini-2.5-flash')

  const [uploading, setUploading] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const job = useJob<PartsResult>()

  function addImg(p: string) {
    const t = p.trim()
    if (t) setImgs(prev => (prev.includes(t) ? prev : [...prev, t]))
  }
  function removeImg(p: string) {
    setImgs(prev => prev.filter(x => x !== p))
  }

  async function onFilesSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    e.target.value = ''  // allow re-selecting the same file later
    if (files.length === 0) return
    setUploading(true)
    try {
      const paths = await Promise.all(files.map(f => uploadFile(f)))
      setImgs(prev => [...prev, ...paths.filter(p => !prev.includes(p))])
    } catch (err) {
      alert(`Upload failed: ${err}`)
    } finally {
      setUploading(false)
    }
  }

  async function handleRun() {
    if (imgs.length === 0) return alert('Add at least one image.')
    if (!apiKey.trim()) return alert('Enter a Gemini API key.')
    await job.run('/api/jobs/parts_json', {
      image_paths:    imgs,
      gemini_api_key: apiKey,
      analyze_model:  analyzeModel,
      grid_cols:      gridCols,
    })
  }

  return (
    <div className="flex flex-col gap-5 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold mb-1">Images → Parts JSON</h2>
        <p className="text-sm text-muted">
          No 3D model needed — upload images directly. The VLM inspects them and
          returns an assembly-tree JSON describing each part (name, color, material).
          When several images are given, they are analyzed together as a grid.
        </p>
      </div>

      <div className="flex flex-col gap-4">
        {/* Image input */}
        <Field label="Images">
          <div className="flex gap-2">
            <TextInput
              placeholder="Type a path + Enter, or browse…"
              value={newImg}
              onChange={e => setNewImg(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') { e.preventDefault(); addImg(newImg); setNewImg('') }
              }}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="px-3 py-2 bg-hover border border-border rounded-lg text-xs text-muted hover:text-white hover:border-accent transition-all whitespace-nowrap disabled:opacity-50"
            >
              {uploading ? 'Uploading…' : 'Browse'}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={onFilesSelected}
            />
          </div>
          {imgs.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {imgs.map(p => (
                <div key={p} className="relative group">
                  <img
                    src={fileUrl(p)}
                    className="rounded-lg border border-border max-h-24 object-contain bg-input"
                    alt="input preview"
                  />
                  <button
                    onClick={() => removeImg(p)}
                    title="Remove"
                    className="absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-bg border border-border text-muted hover:text-white hover:border-accent text-xs leading-none"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </Field>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          <Field label="Gemini API key">
            <TextInput type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="AIza…" />
          </Field>
          <Field label="Analyze model">
            <Select value={analyzeModel} onChange={e => setAnalyzeModel(e.target.value)}>
              {['gemini-2.5-flash','gemini-2.5-pro','gemini-3-flash-preview','gemini-3-pro-preview',
                'gemini-3.1-pro-preview','claude-sonnet-4-6','claude-opus-4-6','claude-haiku-4-5',
                'gpt-4o','gpt-5-mini','gpt-5.2'].map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </Field>
          <Field label="Grid columns (multi-image)">
            <SliderField label="" min={1} max={3} step={1} value={gridCols} onChange={setGridCols} />
          </Field>
        </div>

        <div className="flex items-center gap-3">
          <Btn onClick={handleRun} disabled={job.status === 'running'}>
            Describe Parts
          </Btn>
          <StatusBadge status={job.status} error={job.error} />
        </div>

        {job.result && (
          <div className="bg-bg border border-border rounded-xl overflow-hidden flex flex-col animate-fade-in">
            <div className="px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted flex items-center justify-between">
              <span>Parts JSON</span>
              <button
                onClick={() => downloadJson(job.result!.description, 'parts.json')}
                className="flex items-center gap-1 text-muted hover:text-accent transition-colors"
              >
                <Download size={12} /> Download
              </button>
            </div>
            <pre className="flex-1 p-3 text-[11px] font-mono text-muted overflow-auto bg-input whitespace-pre-wrap break-words max-h-[480px]">
              {JSON.stringify(job.result.description, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}
