/**
 * Shared layout for the three segmentation tabs.
 * Viewers are always rendered; params live in accordions below.
 */
import { useState } from 'react'
import { Viewer3D } from '../Viewer3D'
import { Accordion } from '../ui/Accordion'
import { SamplerFields, useSamplerDefaults } from '../SamplerFields'
import { SplitFields, useSplitDefaults } from '../SplitFields'
import { Btn, StatusBadge } from '../ui/Field'
import { useJob } from '../../hooks/useJob'
import { usePresets } from '../../hooks/usePresets'
import type { SamplerParams } from '../SamplerFields'
import type { SplitParams } from '../SplitFields'

interface Props {
  title?: string
  description?: string
  runEndpoint: string
  buildParams: (sampler: SamplerParams) => Record<string, unknown>
  extraInputs: React.ReactNode
  runLabel: string
}

export function SegTab({ title, description, runEndpoint, buildParams, extraInputs, runLabel }: Props) {
  const { data: presets } = usePresets()
  const [sampler, setSampler] = useState<SamplerParams>(useSamplerDefaults())
  const [split, setSplit]     = useState<SplitParams>(useSplitDefaults())

  const segJob   = useJob<string>()
  const splitJob = useJob<string>()

  async function handleRun() {
    await segJob.run(runEndpoint, buildParams(sampler))
  }

  async function handleSplit() {
    if (!segJob.result) return
    await splitJob.run('/api/jobs/split', {
      in_glb_path: segJob.result,
      ...split,
    })
  }

  return (
    <div className="flex flex-col gap-5 animate-fade-in">
      {/* Header */}
      {(title || description) && (
        <div>
          {title && <h2 className="text-xl font-bold mb-1">{title}</h2>}
          {description && <p className="text-sm text-muted">{description}</p>}
        </div>
      )}

      {/* Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {extraInputs}
      </div>

      {/* Run + Split */}
      <div className="flex items-center gap-3 flex-wrap">
        <Btn onClick={handleRun} disabled={segJob.status === 'running'}>{runLabel}</Btn>
        <StatusBadge status={segJob.status} error={segJob.error} />
        <div className="w-px h-5 bg-border mx-1" />
        <Btn
          variant="secondary"
          onClick={handleSplit}
          disabled={!segJob.result || splitJob.status === 'running'}
        >
          Split into Parts
        </Btn>
        <StatusBadge status={splitJob.status} error={splitJob.error} />
      </div>

      {/* Viewers — always visible */}
      <div className="grid grid-cols-2 gap-4" style={{ minHeight: '360px' }}>
        <Viewer3D label="Segmented Output" filePath={segJob.result} downloadName="segmented.glb" />
        <Viewer3D label="Split Parts Output" filePath={splitJob.result} downloadName="parts.glb" />
      </div>

      {/* Params */}
      <div className="grid grid-cols-2 gap-4">
        <Accordion title="Sampler & Export Parameters">
          <SamplerFields value={sampler} onChange={setSampler} presets={presets?.sampler} />
        </Accordion>
        <Accordion title="Split Parameters">
          <SplitFields value={split} onChange={setSplit} presets={presets?.split} />
        </Accordion>
      </div>
    </div>
  )
}
