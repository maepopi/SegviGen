import { useEffect, useRef, useState } from 'react'
import { Field, TextInput, Select, SliderField, CheckField, Btn, StatusBadge } from '../ui/Field'
import { Accordion } from '../ui/Accordion'
import { SegTab } from './SegTab'
import { useJob } from '../../hooks/useJob'
import { fileUrl, uploadFile, downloadFile } from '../../api/client'
import { Download } from 'lucide-react'
import type { SamplerParams } from '../SamplerFields'

const DEFAULT_CKPT = 'ckpt/full_seg_w_2d_map.ckpt'

const CANONICAL_VIEWS = ['front', 'back', 'left', 'right', 'top', 'bottom'] as const
type View = typeof CANONICAL_VIEWS[number]

interface GuidanceResult { image_path: string; description: unknown }

interface Props {
  glbPath?: string | null
  initialGuidancePath?: string | null
}

export function Full2DTab({ glbPath, initialGuidancePath }: Props) {
  // ── Segmentation GLB path (synced with uploaded model) ────────────────────
  const [segGlb,        setSegGlb]        = useState(glbPath ?? '')
  const [segCkpt,       setSegCkpt]       = useState(DEFAULT_CKPT)

  useEffect(() => { setSegGlb(glbPath ?? '') }, [glbPath])

  // ── Guidance generation state ──────────────────────────────────────────────
  const [glbOverride,   setGlbOverride]   = useState('')
  const [transforms,    setTransforms]    = useState('data_toolkit/transforms.json')
  const [apiKey,        setApiKey]        = useState('')
  const [resolution,    setResolution]    = useState(512)
  const [mode,          setMode]          = useState<'single' | 'grid'>('single')
  const [views,         setViews]         = useState<Record<View, boolean>>({
    front: true, back: true, left: true, right: true, top: false, bottom: false,
  })
  const [gridCols,      setGridCols]      = useState(2)
  const [analyzeModel,  setAnalyzeModel]  = useState('gemini-2.5-flash')
  const [generateModel, setGenerateModel] = useState('gemini-3-pro-image-preview')

  const guidanceJob = useJob<GuidanceResult>()

  // ── Active guidance path (used for segmentation) ───────────────────────────
  const [guidancePath, setGuidancePath] = useState<string | null>(initialGuidancePath ?? null)
  const pathRef    = useRef<HTMLInputElement>(null)
  const previewRef = useRef<HTMLImageElement>(null)

  // Auto-fill when guidance map is generated
  useEffect(() => {
    if (guidanceJob.result) setGuidancePath(guidanceJob.result.image_path)
  }, [guidanceJob.result])

  function toggleView(v: View) {
    setViews(prev => ({ ...prev, [v]: !prev[v] }))
  }

  async function handleGenerateGuidance() {
    const glb = glbOverride.trim() || glbPath || ''
    if (!glb) return alert('Upload a model or enter a GLB path.')
    if (!apiKey.trim()) return alert('Enter a Gemini API key.')
    const selectedViews = CANONICAL_VIEWS.filter(v => views[v])
    if (mode === 'grid' && selectedViews.length === 0) return alert('Select at least one view.')

    await guidanceJob.run('/api/jobs/guidance', {
      glb_path:        glb,
      transforms_path: transforms,
      gemini_api_key:  apiKey,
      analyze_model:   analyzeModel,
      generate_model:  generateModel,
      resolution,
      mode,
      grid_views:      selectedViews,
      grid_cols:       gridCols,
    })
  }

  async function pickGuidance() {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.png,.jpg'
    input.onchange = async () => {
      const file = input.files?.[0]
      if (!file) return
      const serverPath = await uploadFile(file)
      setGuidancePath(serverPath)
      if (previewRef.current) previewRef.current.src = URL.createObjectURL(file)
    }
    input.click()
  }

  // ── Segmentation extraInputs ───────────────────────────────────────────────
  const segInputs = (
    <>
      <Field label="GLB path">
        <TextInput value={segGlb} onChange={e => setSegGlb(e.target.value)}
          placeholder="Leave empty to use uploaded model" />
      </Field>
      <Field label="Checkpoint (.ckpt)">
        <TextInput value={segCkpt} onChange={e => setSegCkpt(e.target.value)} />
      </Field>
      <Field label="2D Guidance Map">
        <div className="flex gap-2">
          <TextInput
            ref={pathRef}
            placeholder="Generate above or browse…"
            value={guidancePath ?? ''}
            onChange={e => setGuidancePath(e.target.value)}
          />
          <button
            onClick={pickGuidance}
            className="px-3 py-2 bg-hover border border-border rounded-lg text-xs text-muted hover:text-white hover:border-accent transition-all whitespace-nowrap"
          >
            Browse
          </button>
        </div>
        {guidancePath && (
          <img
            ref={previewRef}
            src={fileUrl(guidancePath)}
            className="mt-2 rounded-lg border border-border max-h-20 object-contain bg-input"
            alt="guidance preview"
          />
        )}
      </Field>
    </>
  )

  return (
    <div className="flex flex-col gap-5 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold mb-1">Full Segmentation + 2D Guidance Map</h2>
        <p className="text-sm text-muted">
          Optionally generate a flat-color guidance map with Pixmesh, then run segmentation.
        </p>
      </div>

      {/* ── Step 1: Generate guidance map ─────────────────────────────────── */}
      <Accordion title="Step 1 — Generate Guidance Map (Pixmesh)" defaultOpen>
        <div className="flex flex-col gap-4">
          <p className="text-xs text-muted">
            Renders view(s) → VLM describes parts → assigns Kelly palette → image-gen flood-fills → outputs flat-color PNG.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            <Field label="GLB path override (optional)">
              <TextInput value={glbOverride} onChange={e => setGlbOverride(e.target.value)}
                placeholder="Leave empty to use uploaded model" />
            </Field>
            <Field label="Transforms JSON">
              <TextInput value={transforms} onChange={e => setTransforms(e.target.value)} />
            </Field>
            <Field label="Gemini API key">
              <TextInput type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="AIza…" />
            </Field>
            <Field label="Render resolution (px)">
              <SliderField label="" min={256} max={1024} step={128} value={resolution} onChange={setResolution} />
            </Field>
          </div>

          {/* View mode */}
          <div className="flex items-center gap-6">
            <span className="text-xs font-semibold text-muted uppercase tracking-wider">View mode</span>
            {(['single', 'grid'] as const).map(m => (
              <label key={m} className="flex items-center gap-2 cursor-pointer text-sm">
                <input type="radio" name="g-mode" value={m} checked={mode === m}
                  onChange={() => setMode(m)} className="accent-accent" />
                {m === 'single' ? 'Single view' : 'Multi-view grid'}
              </label>
            ))}
          </div>

          {mode === 'grid' && (
            <div className="bg-bg border border-border rounded-xl p-4 space-y-3">
              <div className="flex gap-4 flex-wrap">
                {CANONICAL_VIEWS.map(v => (
                  <CheckField key={v} label={v.charAt(0).toUpperCase() + v.slice(1)}
                    checked={views[v]} onChange={() => toggleView(v)} />
                ))}
              </div>
              <SliderField label="Grid columns" min={1} max={3} step={1} value={gridCols} onChange={setGridCols} />
            </div>
          )}

          <Accordion title="Model Selection" defaultOpen>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Analyze model">
                <Select value={analyzeModel} onChange={e => setAnalyzeModel(e.target.value)}>
                  {['gemini-2.5-flash','gemini-2.5-pro','gemini-3-flash-preview','gemini-3-pro-preview',
                    'gemini-3.1-pro-preview','claude-sonnet-4-6','claude-opus-4-6','claude-haiku-4-5',
                    'gpt-4o','gpt-5-mini','gpt-5.2'].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </Select>
              </Field>
              <Field label="Generate model">
                <Select value={generateModel} onChange={e => setGenerateModel(e.target.value)}>
                  {['gemini-3-pro-image-preview','gemini-3-pro-preview'].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </Select>
              </Field>
            </div>
          </Accordion>

          <div className="flex items-center gap-3">
            <Btn onClick={handleGenerateGuidance} disabled={guidanceJob.status === 'running'}>
              Generate Guidance Map
            </Btn>
            <StatusBadge status={guidanceJob.status} error={guidanceJob.error} />
          </div>

          {guidanceJob.result && (
            <div className="grid grid-cols-2 gap-4 animate-fade-in">
              <div className="bg-bg border border-border rounded-xl overflow-hidden flex flex-col">
                <div className="px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted flex items-center justify-between">
                  <span>Generated Map</span>
                  <button
                    onClick={() => downloadFile(guidanceJob.result!.image_path, 'guidance_map.png')}
                    className="flex items-center gap-1 text-muted hover:text-accent transition-colors"
                  >
                    <Download size={12} /> Download
                  </button>
                </div>
                <img
                  src={fileUrl(guidanceJob.result.image_path)}
                  alt="guidance map"
                  className="w-full object-contain bg-input"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
              <div className="bg-bg border border-border rounded-xl overflow-hidden flex flex-col">
                <div className="px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted">
                  Assembly Tree
                </div>
                <pre className="flex-1 p-3 text-[11px] font-mono text-muted overflow-auto bg-input whitespace-pre-wrap break-words">
                  {JSON.stringify(guidanceJob.result.description, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </Accordion>

      {/* ── Step 2: Run segmentation ───────────────────────────────────────── */}
      <div className="border-t border-border pt-5">
        <p className="text-xs font-semibold uppercase tracking-widest text-dim mb-4">Step 2 — Run Segmentation</p>
        <SegTab
          runEndpoint="/api/jobs/full_2d"
          runLabel="Run 2D-Guided Segmentation"
          buildParams={(sampler: SamplerParams) => ({
            glb_path:     segGlb,
            ckpt_path:    segCkpt,
            guidance_img: guidancePath ?? '',
            ...sampler,
          })}
          extraInputs={segInputs}
        />
      </div>
    </div>
  )
}
