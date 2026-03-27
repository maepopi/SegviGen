import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useCallback } from 'react';
import { uploadFile } from './api/client';
import { FullTab } from './components/tabs/FullTab';
import { Full2DTab } from './components/tabs/Full2DTab';
import { Viewer3D } from './components/Viewer3D';
import { Upload, Grid2x2, Wand2 } from 'lucide-react';
const TABS = [
    { id: 'full', label: 'Full', icon: _jsx(Grid2x2, { size: 15 }) },
    { id: 'full2d', label: 'Full + 2D Map', icon: _jsx(Wand2, { size: 15 }) },
];
export default function App() {
    const [activeTab, setActiveTab] = useState('full');
    const [uploadedPath, setUploadedPath] = useState(null);
    const [uploadedName, setUploadedName] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const handleUpload = useCallback(async (file) => {
        setUploadedName('Uploading…');
        const path = await uploadFile(file);
        setUploadedPath(path);
        setUploadedName(file.name);
    }, []);
    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files[0];
        if (file)
            handleUpload(file);
    }, [handleUpload]);
    const onFileChange = useCallback((e) => {
        const file = e.target.files?.[0];
        if (file)
            handleUpload(file);
    }, [handleUpload]);
    return (_jsxs("div", { className: "flex flex-col h-full bg-bg", children: [_jsxs("header", { className: "flex items-center gap-3 px-5 border-b border-border bg-bg/90 backdrop-blur-sm sticky top-0 z-50 shrink-0", style: { height: '52px' }, children: [_jsx("span", { className: "text-accent font-bold text-lg tracking-tight", children: "SegviGen" }), _jsx("span", { className: "text-dim text-xs", children: "3D Part Segmentation" })] }), _jsxs("div", { className: "flex flex-1 overflow-hidden", children: [_jsxs("aside", { className: "w-64 shrink-0 bg-card border-r border-border flex flex-col gap-4 p-3 overflow-y-auto", children: [_jsxs("div", { children: [_jsx("p", { className: "text-[10px] font-semibold uppercase tracking-widest text-dim mb-2", children: "Input Model" }), _jsxs("label", { onDragOver: e => { e.preventDefault(); setDragOver(true); }, onDragLeave: () => setDragOver(false), onDrop: onDrop, className: `flex items-center gap-3 border-2 border-dashed rounded-xl px-3 py-2 cursor-pointer transition-all
                ${dragOver ? 'border-accent bg-accent/10' : 'border-border hover:border-accent/50 hover:bg-hover'}`, children: [_jsx("input", { type: "file", accept: ".glb,.obj,.ply", className: "hidden", onChange: onFileChange }), _jsx(Upload, { size: 16, className: "text-muted shrink-0" }), _jsx("p", { className: "text-xs text-muted leading-tight truncate", children: uploadedName ?? 'Drop GLB / OBJ / PLY' })] }), _jsx("div", { className: "mt-2 rounded-xl overflow-hidden border border-border", style: { height: '200px' }, children: _jsx(Viewer3D, { filePath: uploadedPath, compact: true }) })] }), _jsxs("div", { children: [_jsx("p", { className: "text-[10px] font-semibold uppercase tracking-widest text-dim mb-2", children: "Mode" }), _jsx("nav", { className: "flex flex-col gap-1", children: TABS.map(tab => (_jsxs("button", { onClick: () => setActiveTab(tab.id), className: `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-left transition-all
                    ${activeTab === tab.id
                                                ? 'bg-accent/10 text-accent font-semibold'
                                                : 'text-muted hover:bg-hover hover:text-white'}`, children: [tab.icon, tab.label] }, tab.id))) })] }), _jsxs("div", { className: "mt-auto text-[11px] text-dim leading-relaxed", children: ["Base models cached on first run.", _jsx("br", {}), "Checkpoints cached per path."] })] }), _jsxs("main", { className: "flex-1 overflow-y-auto p-6", children: [activeTab === 'full' && _jsx(FullTab, { glbPath: uploadedPath }), activeTab === 'full2d' && _jsx(Full2DTab, { glbPath: uploadedPath })] })] })] }));
}
