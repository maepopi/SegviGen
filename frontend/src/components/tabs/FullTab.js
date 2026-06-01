import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useRef, useState, useEffect } from 'react';
import { Field, TextInput } from '../ui/Field';
import { SegTab } from './SegTab';
import { fileUrl, uploadFile } from '../../api/client';
const DEFAULT_TRANSFORMS = 'data_toolkit/transforms.json';
const DEFAULT_CKPT = 'ckpt/full_seg.ckpt';
export function FullTab({ glbPath }) {
    const [glb, setGlb] = useState(glbPath ?? '');
    const [ckpt, setCkpt] = useState(DEFAULT_CKPT);
    const [transforms, setTransforms] = useState(DEFAULT_TRANSFORMS);
    // Optional reference renders. Empty → backend auto-renders one view.
    // The model conditions on every image at once, so any number works.
    const [imgs, setImgs] = useState([]);
    const [newImg, setNewImg] = useState('');
    const [uploading, setUploading] = useState(false);
    const fileRef = useRef(null);
    useEffect(() => { setGlb(glbPath ?? ''); }, [glbPath]);
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
        e.target.value = '';
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
    return (_jsx(SegTab, { title: "Full Segmentation", description: "Automatically segments all parts simultaneously, conditioned on one or more rendered views of the model.", runEndpoint: "/api/jobs/full", runLabel: "Run Full Segmentation", buildParams: (sampler) => ({
            glb_path: glb,
            ckpt_path: ckpt,
            transforms_path: transforms,
            rendered_img: imgs.length ? imgs : null,
            ...sampler,
        }), extraInputs: _jsxs(_Fragment, { children: [_jsx(Field, { label: "GLB path", children: _jsx(TextInput, { value: glb, onChange: e => setGlb(e.target.value), placeholder: "Leave empty to use uploaded model" }) }), _jsx(Field, { label: "Checkpoint (.ckpt)", children: _jsx(TextInput, { value: ckpt, onChange: e => setCkpt(e.target.value) }) }), _jsx(Field, { label: "Transforms JSON", children: _jsx(TextInput, { value: transforms, onChange: e => setTransforms(e.target.value) }) }), _jsxs(Field, { label: "Override rendered image(s) (optional)", children: [_jsxs("div", { className: "flex gap-2", children: [_jsx(TextInput, { value: newImg, onChange: e => setNewImg(e.target.value), placeholder: "path/to/image.png \u2014 type + Enter, or browse\u2026", onKeyDown: e => {
                                        if (e.key === 'Enter') {
                                            e.preventDefault();
                                            addImg(newImg);
                                            setNewImg('');
                                        }
                                    } }), _jsx("button", { onClick: () => fileRef.current?.click(), disabled: uploading, className: "px-3 py-2 bg-hover border border-border rounded-lg text-xs text-muted hover:text-white hover:border-accent transition-all whitespace-nowrap disabled:opacity-50", children: uploading ? 'Uploading…' : 'Browse' }), _jsx("input", { ref: fileRef, type: "file", accept: "image/*", multiple: true, className: "hidden", onChange: onFilesSelected })] }), imgs.length > 0 && (_jsx("div", { className: "mt-2 flex flex-wrap gap-2", children: imgs.map(p => (_jsxs("div", { className: "relative group", children: [_jsx("img", { src: fileUrl(p), className: "rounded-lg border border-border max-h-20 object-contain bg-input", alt: "render preview" }), _jsx("button", { onClick: () => removeImg(p), title: "Remove", className: "absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-bg border border-border text-muted hover:text-white hover:border-accent text-xs leading-none", children: "\u00D7" })] }, p))) }))] })] }) }));
}
