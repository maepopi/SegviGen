import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useRef, useState } from 'react';
import { Field, TextInput, Select, SliderField, CheckField, Btn, StatusBadge } from '../ui/Field';
import { Accordion } from '../ui/Accordion';
import { SegTab } from './SegTab';
import { useJob } from '../../hooks/useJob';
import { fileUrl, uploadFile, downloadFile } from '../../api/client';
import { Download } from 'lucide-react';
const DEFAULT_CKPT = 'ckpt/full_seg_w_2d_map.ckpt';
const CANONICAL_VIEWS = ['front', 'back', 'left', 'right', 'top', 'bottom'];
export function Full2DTab({ glbPath, initialGuidancePath }) {
    // ── Segmentation GLB path (synced with uploaded model) ────────────────────
    const [segGlb, setSegGlb] = useState(glbPath ?? '');
    const [segCkpt, setSegCkpt] = useState(DEFAULT_CKPT);
    useEffect(() => { setSegGlb(glbPath ?? ''); }, [glbPath]);
    // ── Guidance generation state ──────────────────────────────────────────────
    const [glbOverride, setGlbOverride] = useState('');
    const [transforms, setTransforms] = useState('data_toolkit/transforms.json');
    const [apiKey, setApiKey] = useState('');
    const [resolution, setResolution] = useState(512);
    const [mode, setMode] = useState('single');
    const [views, setViews] = useState({
        front: true, back: true, left: true, right: true, top: false, bottom: false,
    });
    const [gridCols, setGridCols] = useState(2);
    const [analyzeModel, setAnalyzeModel] = useState('gemini-2.5-flash');
    const [generateModel, setGenerateModel] = useState('gemini-3-pro-image-preview');
    const guidanceJob = useJob();
    // ── Active guidance path (used for segmentation) ───────────────────────────
    const [guidancePath, setGuidancePath] = useState(initialGuidancePath ?? null);
    const pathRef = useRef(null);
    const previewRef = useRef(null);
    // Auto-fill when guidance map is generated
    useEffect(() => {
        if (guidanceJob.result)
            setGuidancePath(guidanceJob.result.image_path);
    }, [guidanceJob.result]);
    function toggleView(v) {
        setViews(prev => ({ ...prev, [v]: !prev[v] }));
    }
    async function handleGenerateGuidance() {
        const glb = glbOverride.trim() || glbPath || '';
        if (!glb)
            return alert('Upload a model or enter a GLB path.');
        if (!apiKey.trim())
            return alert('Enter a Gemini API key.');
        const selectedViews = CANONICAL_VIEWS.filter(v => views[v]);
        if (mode === 'grid' && selectedViews.length === 0)
            return alert('Select at least one view.');
        await guidanceJob.run('/api/jobs/guidance', {
            glb_path: glb,
            transforms_path: transforms,
            gemini_api_key: apiKey,
            analyze_model: analyzeModel,
            generate_model: generateModel,
            resolution,
            mode,
            grid_views: selectedViews,
            grid_cols: gridCols,
        });
    }
    async function pickGuidance() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.png,.jpg';
        input.onchange = async () => {
            const file = input.files?.[0];
            if (!file)
                return;
            const serverPath = await uploadFile(file);
            setGuidancePath(serverPath);
            if (previewRef.current)
                previewRef.current.src = URL.createObjectURL(file);
        };
        input.click();
    }
    // ── Segmentation extraInputs ───────────────────────────────────────────────
    const segInputs = (_jsxs(_Fragment, { children: [_jsx(Field, { label: "GLB path", children: _jsx(TextInput, { value: segGlb, onChange: e => setSegGlb(e.target.value), placeholder: "Leave empty to use uploaded model" }) }), _jsx(Field, { label: "Checkpoint (.ckpt)", children: _jsx(TextInput, { value: segCkpt, onChange: e => setSegCkpt(e.target.value) }) }), _jsxs(Field, { label: "2D Guidance Map", children: [_jsxs("div", { className: "flex gap-2", children: [_jsx(TextInput, { ref: pathRef, placeholder: "Generate above or browse\u2026", value: guidancePath ?? '', onChange: e => setGuidancePath(e.target.value) }), _jsx("button", { onClick: pickGuidance, className: "px-3 py-2 bg-hover border border-border rounded-lg text-xs text-muted hover:text-white hover:border-accent transition-all whitespace-nowrap", children: "Browse" })] }), guidancePath && (_jsx("img", { ref: previewRef, src: fileUrl(guidancePath), className: "mt-2 rounded-lg border border-border max-h-20 object-contain bg-input", alt: "guidance preview" }))] })] }));
    return (_jsxs("div", { className: "flex flex-col gap-5 animate-fade-in", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-bold mb-1", children: "Full Segmentation + 2D Guidance Map" }), _jsx("p", { className: "text-sm text-muted", children: "Optionally generate a flat-color guidance map with Pixmesh, then run segmentation." })] }), _jsx(Accordion, { title: "Step 1 \u2014 Generate Guidance Map (Pixmesh)", defaultOpen: true, children: _jsxs("div", { className: "flex flex-col gap-4", children: [_jsx("p", { className: "text-xs text-muted", children: "Renders view(s) \u2192 VLM describes parts \u2192 assigns Kelly palette \u2192 image-gen flood-fills \u2192 outputs flat-color PNG." }), _jsxs("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3", children: [_jsx(Field, { label: "GLB path override (optional)", children: _jsx(TextInput, { value: glbOverride, onChange: e => setGlbOverride(e.target.value), placeholder: "Leave empty to use uploaded model" }) }), _jsx(Field, { label: "Transforms JSON", children: _jsx(TextInput, { value: transforms, onChange: e => setTransforms(e.target.value) }) }), _jsx(Field, { label: "Gemini API key", children: _jsx(TextInput, { type: "password", value: apiKey, onChange: e => setApiKey(e.target.value), placeholder: "AIza\u2026" }) }), _jsx(Field, { label: "Render resolution (px)", children: _jsx(SliderField, { label: "", min: 256, max: 1024, step: 128, value: resolution, onChange: setResolution }) })] }), _jsxs("div", { className: "flex items-center gap-6", children: [_jsx("span", { className: "text-xs font-semibold text-muted uppercase tracking-wider", children: "View mode" }), ['single', 'grid'].map(m => (_jsxs("label", { className: "flex items-center gap-2 cursor-pointer text-sm", children: [_jsx("input", { type: "radio", name: "g-mode", value: m, checked: mode === m, onChange: () => setMode(m), className: "accent-accent" }), m === 'single' ? 'Single view' : 'Multi-view grid'] }, m)))] }), mode === 'grid' && (_jsxs("div", { className: "bg-bg border border-border rounded-xl p-4 space-y-3", children: [_jsx("div", { className: "flex gap-4 flex-wrap", children: CANONICAL_VIEWS.map(v => (_jsx(CheckField, { label: v.charAt(0).toUpperCase() + v.slice(1), checked: views[v], onChange: () => toggleView(v) }, v))) }), _jsx(SliderField, { label: "Grid columns", min: 1, max: 3, step: 1, value: gridCols, onChange: setGridCols })] })), _jsx(Accordion, { title: "Model Selection", defaultOpen: true, children: _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsx(Field, { label: "Analyze model", children: _jsx(Select, { value: analyzeModel, onChange: e => setAnalyzeModel(e.target.value), children: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-flash-preview', 'gemini-3-pro-preview',
                                                'gemini-3.1-pro-preview', 'claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5',
                                                'gpt-4o', 'gpt-5-mini', 'gpt-5.2'].map(m => (_jsx("option", { value: m, children: m }, m))) }) }), _jsx(Field, { label: "Generate model", children: _jsx(Select, { value: generateModel, onChange: e => setGenerateModel(e.target.value), children: ['gemini-3-pro-image-preview', 'gemini-3-pro-preview'].map(m => (_jsx("option", { value: m, children: m }, m))) }) })] }) }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx(Btn, { onClick: handleGenerateGuidance, disabled: guidanceJob.status === 'running', children: "Generate Guidance Map" }), _jsx(StatusBadge, { status: guidanceJob.status, error: guidanceJob.error })] }), guidanceJob.result && (_jsxs("div", { className: "grid grid-cols-2 gap-4 animate-fade-in", children: [_jsxs("div", { className: "bg-bg border border-border rounded-xl overflow-hidden flex flex-col", children: [_jsxs("div", { className: "px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted flex items-center justify-between", children: [_jsx("span", { children: "Generated Map" }), _jsxs("button", { onClick: () => downloadFile(guidanceJob.result.image_path, 'guidance_map.png'), className: "flex items-center gap-1 text-muted hover:text-accent transition-colors", children: [_jsx(Download, { size: 12 }), " Download"] })] }), _jsx("img", { src: fileUrl(guidanceJob.result.image_path), alt: "guidance map", className: "w-full object-contain bg-input", style: { imageRendering: 'pixelated' } })] }), _jsxs("div", { className: "bg-bg border border-border rounded-xl overflow-hidden flex flex-col", children: [_jsx("div", { className: "px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted", children: "Assembly Tree" }), _jsx("pre", { className: "flex-1 p-3 text-[11px] font-mono text-muted overflow-auto bg-input whitespace-pre-wrap break-words", children: JSON.stringify(guidanceJob.result.description, null, 2) })] })] }))] }) }), _jsxs("div", { className: "border-t border-border pt-5", children: [_jsx("p", { className: "text-xs font-semibold uppercase tracking-widest text-dim mb-4", children: "Step 2 \u2014 Run Segmentation" }), _jsx(SegTab, { runEndpoint: "/api/jobs/full_2d", runLabel: "Run 2D-Guided Segmentation", buildParams: (sampler) => ({
                            glb_path: segGlb,
                            ckpt_path: segCkpt,
                            guidance_img: guidancePath ?? '',
                            ...sampler,
                        }), extraInputs: segInputs })] })] }));
}
