import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Images → Parts JSON.
 * A model-free mode: upload one or more images and run the VLM analysis to get
 * back the assembly-tree JSON describing each part (name, color, material).
 * No 3D model and no guidance-map image — analysis only.
 */
import { useRef, useState } from 'react';
import { Field, TextInput, Select, SliderField, Btn, StatusBadge } from '../ui/Field';
import { useJob } from '../../hooks/useJob';
import { fileUrl, uploadFile } from '../../api/client';
import { Download } from 'lucide-react';
function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}
export function ImagesTab() {
    const [imgs, setImgs] = useState([]);
    const [newImg, setNewImg] = useState('');
    const [apiKey, setApiKey] = useState('');
    const [gridCols, setGridCols] = useState(2);
    const [analyzeModel, setAnalyzeModel] = useState('gemini-2.5-flash');
    const [uploading, setUploading] = useState(false);
    const fileRef = useRef(null);
    const job = useJob();
    function addImg(p) {
        const t = p.trim();
        if (t)
            setImgs(prev => (prev.includes(t) ? prev : [...prev, t]));
    }
    function removeImg(p) {
        setImgs(prev => prev.filter(x => x !== p));
    }
    async function onFilesSelected(e) {
        const files = Array.from(e.target.files ?? []);
        e.target.value = ''; // allow re-selecting the same file later
        if (files.length === 0)
            return;
        setUploading(true);
        try {
            const paths = await Promise.all(files.map(f => uploadFile(f)));
            setImgs(prev => [...prev, ...paths.filter(p => !prev.includes(p))]);
        }
        catch (err) {
            alert(`Upload failed: ${err}`);
        }
        finally {
            setUploading(false);
        }
    }
    async function handleRun() {
        if (imgs.length === 0)
            return alert('Add at least one image.');
        if (!apiKey.trim())
            return alert('Enter a Gemini API key.');
        await job.run('/api/jobs/parts_json', {
            image_paths: imgs,
            gemini_api_key: apiKey,
            analyze_model: analyzeModel,
            grid_cols: gridCols,
        });
    }
    return (_jsxs("div", { className: "flex flex-col gap-5 animate-fade-in", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-xl font-bold mb-1", children: "Images \u2192 Parts JSON" }), _jsx("p", { className: "text-sm text-muted", children: "No 3D model needed \u2014 upload images directly. The VLM inspects them and returns an assembly-tree JSON describing each part (name, color, material). When several images are given, they are analyzed together as a grid." })] }), _jsxs("div", { className: "flex flex-col gap-4", children: [_jsxs(Field, { label: "Images", children: [_jsxs("div", { className: "flex gap-2", children: [_jsx(TextInput, { placeholder: "Type a path + Enter, or browse\u2026", value: newImg, onChange: e => setNewImg(e.target.value), onKeyDown: e => {
                                            if (e.key === 'Enter') {
                                                e.preventDefault();
                                                addImg(newImg);
                                                setNewImg('');
                                            }
                                        } }), _jsx("button", { onClick: () => fileRef.current?.click(), disabled: uploading, className: "px-3 py-2 bg-hover border border-border rounded-lg text-xs text-muted hover:text-white hover:border-accent transition-all whitespace-nowrap disabled:opacity-50", children: uploading ? 'Uploading…' : 'Browse' }), _jsx("input", { ref: fileRef, type: "file", accept: "image/*", multiple: true, className: "hidden", onChange: onFilesSelected })] }), imgs.length > 0 && (_jsx("div", { className: "mt-2 flex flex-wrap gap-2", children: imgs.map(p => (_jsxs("div", { className: "relative group", children: [_jsx("img", { src: fileUrl(p), className: "rounded-lg border border-border max-h-24 object-contain bg-input", alt: "input preview" }), _jsx("button", { onClick: () => removeImg(p), title: "Remove", className: "absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-bg border border-border text-muted hover:text-white hover:border-accent text-xs leading-none", children: "\u00D7" })] }, p))) }))] }), _jsxs("div", { className: "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3", children: [_jsx(Field, { label: "Gemini API key", children: _jsx(TextInput, { type: "password", value: apiKey, onChange: e => setApiKey(e.target.value), placeholder: "AIza\u2026" }) }), _jsx(Field, { label: "Analyze model", children: _jsx(Select, { value: analyzeModel, onChange: e => setAnalyzeModel(e.target.value), children: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-flash-preview', 'gemini-3-pro-preview',
                                        'gemini-3.1-pro-preview', 'claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5',
                                        'gpt-4o', 'gpt-5-mini', 'gpt-5.2'].map(m => (_jsx("option", { value: m, children: m }, m))) }) }), _jsx(Field, { label: "Grid columns (multi-image)", children: _jsx(SliderField, { label: "", min: 1, max: 3, step: 1, value: gridCols, onChange: setGridCols }) })] }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx(Btn, { onClick: handleRun, disabled: job.status === 'running', children: "Describe Parts" }), _jsx(StatusBadge, { status: job.status, error: job.error })] }), job.result && (_jsxs("div", { className: "bg-bg border border-border rounded-xl overflow-hidden flex flex-col animate-fade-in", children: [_jsxs("div", { className: "px-3 py-2 border-b border-border text-xs font-semibold uppercase tracking-wider text-muted flex items-center justify-between", children: [_jsx("span", { children: "Parts JSON" }), _jsxs("button", { onClick: () => downloadJson(job.result.description, 'parts.json'), className: "flex items-center gap-1 text-muted hover:text-accent transition-colors", children: [_jsx(Download, { size: 12 }), " Download"] })] }), _jsx("pre", { className: "flex-1 p-3 text-[11px] font-mono text-muted overflow-auto bg-input whitespace-pre-wrap break-words max-h-[480px]", children: JSON.stringify(job.result.description, null, 2) })] }))] })] }));
}
