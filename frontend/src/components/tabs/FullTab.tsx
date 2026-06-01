import { useRef, useState, useEffect } from 'react'
import { Field, TextInput } from '../ui/Field'
import { SegTab } from './SegTab'
import { fileUrl, uploadFile } from '../../api/client'
import type { SamplerParams } from '../SamplerFields'

const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json'
const DEFAULT_CKPT       = 'ckpt/full_seg.ckpt'

interface Props { glbPath?: string | null }

export function FullTab({ glbPath }: Props) {
  const [glb,        setGlb]        = useState(glbPath ?? '')
  const [ckpt,       setCkpt]       = useState(DEFAULT_CKPT)
  const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS)
  // Optional reference renders. Empty → backend auto-renders one view.
  // The model conditions on every image at once, so any number works.
  const [imgs,       setImgs]       = useState<string[]>([])
  const [newImg,     setNewImg]     = useState('')
  const [uploading,  setUploading]  = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { setGlb(glbPath ?? '') }, [glbPath])

  function addImg(p: string) {
    const t = p.trim()
    if (t) setImgs(prev => (prev.includes(t) ? prev : [...prev, t]))
  }
  function removeImg(p: string) {
    setImgs(prev => prev.filter(x => x !== p))
  }

  async function onFilesSelected(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    e.target.value = ''
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

  return (
    <SegTab
      title="Full Segmentation"
      description="Automatically segments all parts simultaneously, conditioned on one or more rendered views of the model."
      runEndpoint="/api/jobs/full"
      runLabel="Run Full Segmentation"
      buildParams={(sampler: SamplerParams) => ({
        glb_path:        glb,
        ckpt_path:       ckpt,
        transforms_path: transforms,
        rendered_img:    imgs.length ? imgs : null,
        ...sampler,
      })}
      extraInputs={
        <>
          <Field label="GLB path">
            <TextInput value={glb} onChange={e => setGlb(e.target.value)}
              placeholder="Leave empty to use uploaded model" />
          </Field>
          <Field label="Checkpoint (.ckpt)">
            <TextInput value={ckpt} onChange={e => setCkpt(e.target.value)} />
          </Field>
          <Field label="Transforms JSON">
            <TextInput value={transforms} onChange={e => setTransforms(e.target.value)} />
          </Field>
          <Field label="Override rendered image(s) (optional)">
            <div className="flex gap-2">
              <TextInput value={newImg} onChange={e => setNewImg(e.target.value)}
                placeholder="path/to/image.png — type + Enter, or browse…"
                onKeyDown={e => {
                  if (e.key === 'Enter') { e.preventDefault(); addImg(newImg); setNewImg('') }
                }} />
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
                      className="rounded-lg border border-border max-h-20 object-contain bg-input"
                      alt="render preview"
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
        </>
      }
    />
  )
}
