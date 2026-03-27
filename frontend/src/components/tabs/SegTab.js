import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Shared layout for the three segmentation tabs.
 * Viewers are always rendered; params live in accordions below.
 */
import { useState } from 'react';
import { Viewer3D } from '../Viewer3D';
import { Accordion } from '../ui/Accordion';
import { SamplerFields, useSamplerDefaults } from '../SamplerFields';
import { SplitFields, useSplitDefaults } from '../SplitFields';
import { Btn, StatusBadge } from '../ui/Field';
import { useJob } from '../../hooks/useJob';
import { usePresets } from '../../hooks/usePresets';
export function SegTab({ title, description, runEndpoint, buildParams, extraInputs, runLabel }) {
    const { data: presets } = usePresets();
    const [sampler, setSampler] = useState(useSamplerDefaults());
    const [split, setSplit] = useState(useSplitDefaults());
    const segJob = useJob();
    const splitJob = useJob();
    async function handleRun() {
        await segJob.run(runEndpoint, buildParams(sampler));
    }
    async function handleSplit() {
        if (!segJob.result)
            return;
        await splitJob.run('/api/jobs/split', {
            in_glb_path: segJob.result,
            ...split,
        });
    }
    return (_jsxs("div", { className: "flex flex-col gap-5 animate-fade-in", children: [(title || description) && (_jsxs("div", { children: [title && _jsx("h2", { className: "text-xl font-bold mb-1", children: title }), description && _jsx("p", { className: "text-sm text-muted", children: description })] })), _jsx("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3", children: extraInputs }), _jsxs("div", { className: "flex items-center gap-3 flex-wrap", children: [_jsx(Btn, { onClick: handleRun, disabled: segJob.status === 'running', children: runLabel }), _jsx(StatusBadge, { status: segJob.status, error: segJob.error }), _jsx("div", { className: "w-px h-5 bg-border mx-1" }), _jsx(Btn, { variant: "secondary", onClick: handleSplit, disabled: !segJob.result || splitJob.status === 'running', children: "Split into Parts" }), _jsx(StatusBadge, { status: splitJob.status, error: splitJob.error })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", style: { minHeight: '360px' }, children: [_jsx(Viewer3D, { label: "Segmented Output", filePath: segJob.result, downloadName: "segmented.glb" }), _jsx(Viewer3D, { label: "Split Parts Output", filePath: splitJob.result, downloadName: "parts.glb" })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsx(Accordion, { title: "Sampler & Export Parameters", children: _jsx(SamplerFields, { value: sampler, onChange: setSampler, presets: presets?.sampler }) }), _jsx(Accordion, { title: "Split Parameters", children: _jsx(SplitFields, { value: split, onChange: setSplit, presets: presets?.split }) })] })] }));
}
