"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { 
  UploadCloud, 
  Sparkles, 
  Download, 
  Trash2, 
  CheckCircle2, 
  AlertCircle, 
  Loader2, 
  Archive,
  Maximize2,
  X,
  Key,
  ShieldCheck,
  HelpCircle
} from "lucide-react";
import JSZip from "jszip";
import { saveAs } from "file-saver";

interface BatchItem {
  id: string;
  file: File;
  name: string;
  previewUrl: string;
  status: "idle" | "processing" | "completed" | "error";
  error?: string;
  progress: number;
  r2Url?: string;
  upscaledBase64?: string;
  originalWidth?: number;
  originalHeight?: number;
  upscaledWidth?: number;
  upscaledHeight?: number;
}

const LOCAL_KEY_STORAGE = "batch_upscaler_runpod_key";
const LOCAL_ENDPOINT_STORAGE = "batch_upscaler_endpoint_id";

export default function Home() {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [scale, setScale] = useState<number>(4);
  const [faceEnhance, setFaceEnhance] = useState<boolean>(false);
  const [removeBg, setRemoveBg] = useState<boolean>(false);
  const [isBatchProcessing, setIsBatchProcessing] = useState<boolean>(false);
  const [isZipping, setIsZipping] = useState<boolean>(false);
  const [previewModalItem, setPreviewModalItem] = useState<BatchItem | null>(null);
  
  // BYOK (Bring Your Own Key) States
  const [userApiKey, setUserApiKey] = useState<string>("");
  const [userEndpointId, setUserEndpointId] = useState<string>("3j67gpfsvuwgy3");
  const [showKeyModal, setShowKeyModal] = useState<boolean>(false);
  const [keyInputTemp, setKeyInputTemp] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load API key from localStorage on mount
  useEffect(() => {
    try {
      const savedKey = localStorage.getItem(LOCAL_KEY_STORAGE);
      const savedEndpoint = localStorage.getItem(LOCAL_ENDPOINT_STORAGE);
      if (savedKey) {
        setUserApiKey(savedKey);
        setKeyInputTemp(savedKey);
      }
      if (savedEndpoint) {
        setUserEndpointId(savedEndpoint);
      }
    } catch (e) {
      console.warn("Could not load from localStorage:", e);
    }
  }, []);

  const handleSaveApiKey = () => {
    const trimmed = keyInputTemp.trim();
    setUserApiKey(trimmed);
    try {
      if (trimmed) {
        localStorage.setItem(LOCAL_KEY_STORAGE, trimmed);
      } else {
        localStorage.removeItem(LOCAL_KEY_STORAGE);
      }
    } catch (e) {
      console.warn("Could not save to localStorage:", e);
    }
    setShowKeyModal(false);
  };

  const addFiles = useCallback((files: FileList | File[]) => {
    const newFiles = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (newFiles.length === 0) return;

    const newItems: BatchItem[] = newFiles.map((file) => {
      const previewUrl = URL.createObjectURL(file);
      const item: BatchItem = {
        id: Math.random().toString(36).substring(2, 11),
        file,
        name: file.name,
        previewUrl,
        status: "idle",
        progress: 0,
      };

      const img = new Image();
      img.src = previewUrl;
      img.onload = () => {
        setItems((prev) =>
          prev.map((it) =>
            it.id === item.id
              ? { ...it, originalWidth: img.naturalWidth, originalHeight: img.naturalHeight }
              : it
          )
        );
      };

      return item;
    });

    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      addFiles(e.dataTransfer.files);
    }
  };

  const handleProcessItem = async (item: BatchItem) => {
    try {
      setItems((prev) =>
        prev.map((it) => (it.id === item.id ? { ...it, status: "processing", progress: 25 } : it))
      );

      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
      });
      reader.readAsDataURL(item.file);
      const base64 = await base64Promise;

      setItems((prev) =>
        prev.map((it) => (it.id === item.id ? { ...it, progress: 50 } : it))
      );

      const res = await fetch("/api/upscale", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64,
          scale,
          face_enhance: faceEnhance,
          remove_bg: removeBg,
          model: scale === 2 ? "x2plus" : "x4plus",
          userApiKey: userApiKey.trim(),
          userEndpointId: userEndpointId.trim(),
        }),
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        if (res.status === 401) {
          setShowKeyModal(true);
        }
        throw new Error(data.error || "Upscale failed");
      }

      setItems((prev) =>
        prev.map((it) =>
          it.id === item.id
            ? {
                ...it,
                status: "completed",
                progress: 100,
                r2Url: data.r2Url,
                upscaledWidth: data.width,
                upscaledHeight: data.height,
              }
            : it
        )
      );
    } catch (err: any) {
      setItems((prev) =>
        prev.map((it) =>
          it.id === item.id ? { ...it, status: "error", error: err.message || "Failed" } : it
        )
      );
    }
  };

  const handleStartBatch = async () => {
    if (!userApiKey.trim()) {
      setShowKeyModal(true);
      return;
    }

    const queue = items.filter((it) => it.status === "idle" || it.status === "error");
    if (queue.length === 0) return;

    setIsBatchProcessing(true);

    const CONCURRENCY = 5;
    const running = [...queue];

    const worker = async () => {
      while (running.length > 0) {
        const item = running.shift();
        if (item) {
          await handleProcessItem(item);
        }
      }
    };

    await Promise.all(Array.from({ length: CONCURRENCY }).map(() => worker()));
    setIsBatchProcessing(false);
  };

  const handleDownloadAllZip = async () => {
    const completed = items.filter((it) => it.status === "completed" && (it.r2Url || it.upscaledBase64));
    if (completed.length === 0) return;

    setIsZipping(true);
    try {
      const zip = new JSZip();

      await Promise.all(
        completed.map(async (it) => {
          const ext = removeBg ? "png" : "jpg";
          const dotIdx = it.name.lastIndexOf(".");
          const rawName = dotIdx !== -1 ? it.name.substring(0, dotIdx) : it.name;
          const filename = `${rawName}_${scale}x.${ext}`;

          if (it.r2Url) {
            const proxyUrl = `/api/download?url=${encodeURIComponent(it.r2Url)}&filename=${encodeURIComponent(filename)}`;
            const resp = await fetch(proxyUrl);
            const blob = await resp.blob();
            zip.file(filename, blob);
          } else if (it.upscaledBase64) {
            const base64Data = it.upscaledBase64.split(",")[1] || it.upscaledBase64;
            zip.file(filename, base64Data, { base64: true });
          }
        })
      );

      const zipBlob = await zip.generateAsync({ type: "blob" });
      saveAs(zipBlob, `batch_upscaled_${Date.now()}.zip`);
    } catch (e) {
      console.error("ZIP Error:", e);
    } finally {
      setIsZipping(false);
    }
  };

  const handleDownloadSingle = async (item: BatchItem) => {
    const ext = removeBg ? "png" : "jpg";
    const dotIdx = item.name.lastIndexOf(".");
    const rawName = dotIdx !== -1 ? item.name.substring(0, dotIdx) : item.name;
    const filename = `${rawName}_${scale}x.${ext}`;

    if (item.r2Url) {
      const proxyUrl = `/api/download?url=${encodeURIComponent(item.r2Url)}&filename=${encodeURIComponent(filename)}`;
      const link = document.createElement("a");
      link.href = proxyUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else if (item.upscaledBase64) {
      saveAs(item.upscaledBase64, filename);
    }
  };

  const handleRemoveItem = (id: string) => {
    setItems((prev) => prev.filter((it) => it.id !== id));
  };

  const handleClearAll = () => {
    setItems([]);
  };

  const completedCount = items.filter((it) => it.status === "completed").length;

  return (
    <div className="min-h-screen bg-[#09090b] text-neutral-100 flex flex-col font-sans selection:bg-cyan-500/30">
      {/* Header */}
      <header className="h-16 border-b border-neutral-800/80 px-6 flex items-center justify-between bg-[#09090b]/80 backdrop-blur sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <span className="text-sm font-semibold tracking-tight text-white">Batch Upscaler</span>
            <span className="text-[11px] text-neutral-400 ml-2 font-mono bg-neutral-800/60 px-1.5 py-0.5 rounded">RunPod 4K Turbo</span>
          </div>
        </div>

        {/* Global Controls & Social */}
        <div className="flex items-center gap-3">
          <div className="flex items-center bg-neutral-900 border border-neutral-800 rounded-lg p-0.5 text-xs">
            <button
              onClick={() => setScale(2)}
              className={
                scale === 2
                  ? "px-2.5 py-1 rounded-md font-medium transition bg-neutral-800 text-cyan-400 shadow-sm"
                  : "px-2.5 py-1 rounded-md font-medium transition text-neutral-400 hover:text-white"
              }
            >
              2x
            </button>
            <button
              onClick={() => setScale(4)}
              className={
                scale === 4
                  ? "px-2.5 py-1 rounded-md font-medium transition bg-neutral-800 text-cyan-400 shadow-sm"
                  : "px-2.5 py-1 rounded-md font-medium transition text-neutral-400 hover:text-white"
              }
            >
              4x UHD
            </button>
          </div>

          <label className="flex items-center gap-1.5 text-xs text-neutral-300 cursor-pointer select-none bg-neutral-900 border border-neutral-800 px-2.5 py-1 rounded-lg hover:bg-neutral-800/80 transition">
            <input
              type="checkbox"
              checked={faceEnhance}
              onChange={(e) => setFaceEnhance(e.target.checked)}
              className="rounded accent-cyan-500 w-3.5 h-3.5"
            />
            Face Fix
          </label>

          <label className="flex items-center gap-1.5 text-xs text-neutral-300 cursor-pointer select-none bg-neutral-900 border border-neutral-800 px-2.5 py-1 rounded-lg hover:bg-neutral-800/80 transition">
            <input
              type="checkbox"
              checked={removeBg}
              onChange={(e) => setRemoveBg(e.target.checked)}
              className="rounded accent-cyan-500 w-3.5 h-3.5"
            />
            Remove BG
          </label>

          {/* BYOK API Key Button */}
          <button
            onClick={() => setShowKeyModal(true)}
            className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-lg border transition ${
              userApiKey.trim()
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
                : "bg-amber-500/10 border-amber-500/30 text-amber-300 hover:bg-amber-500/20 animate-pulse"
            }`}
          >
            <Key className="w-3.5 h-3.5" />
            <span>{userApiKey.trim() ? "Key Connected" : "Set API Key"}</span>
          </button>

          <a
            href="https://github.com/effame/batch-upscaler"
            target="_blank"
            rel="noreferrer"
            className="p-1.5 rounded-lg border border-neutral-800 hover:bg-neutral-800 text-neutral-400 hover:text-white transition"
            title="GitHub Repository"
          >
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-5xl w-full mx-auto p-6 flex flex-col gap-5">
        {/* Clean Minimal Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border border-dashed border-neutral-800 hover:border-cyan-500/50 bg-neutral-900/20 hover:bg-neutral-900/50 transition-all rounded-2xl p-10 flex flex-col items-center justify-center gap-2.5 cursor-pointer group"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files && addFiles(e.target.files)}
            multiple
            accept="image/*"
            className="hidden"
          />
          <div className="w-11 h-11 rounded-xl bg-neutral-800/80 group-hover:bg-cyan-500/10 group-hover:text-cyan-400 flex items-center justify-center transition text-neutral-400">
            <UploadCloud className="w-5 h-5" />
          </div>
          <div className="text-center">
            <p className="text-xs font-medium text-neutral-200">
              ลากรูปภาพมาวางที่นี่ หรือ <span className="text-cyan-400 hover:underline">คลิกเพื่อเลือกไฟล์</span>
            </p>
            <p className="text-[11px] text-neutral-400 mt-0.5">
              รองรับทีละหลายรูปพร้อมกัน (10-50+ ภาพ) • JPG, PNG, WebP
            </p>
          </div>
        </div>

        {/* Action Header Bar */}
        {items.length > 0 && (
          <div className="flex items-center justify-between bg-neutral-900/60 border border-neutral-800/80 rounded-xl px-4 py-2.5">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-neutral-300">
                {items.length} รายการ
              </span>
              <span className="text-neutral-700 text-xs">/</span>
              <span className="text-xs text-emerald-400 font-medium">
                เสร็จแล้ว {completedCount}
              </span>
              <button
                onClick={handleClearAll}
                className="text-[11px] text-neutral-400 hover:text-rose-400 ml-2 transition"
              >
                ล้างทั้งหมด
              </button>
            </div>

            <div className="flex items-center gap-2">
              {completedCount > 0 && (
                <button
                  onClick={handleDownloadAllZip}
                  disabled={isZipping}
                  className="flex items-center gap-1.5 bg-neutral-800 hover:bg-neutral-700 text-white text-xs font-medium px-3 py-1.5 rounded-lg transition"
                >
                  {isZipping ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Archive className="w-3.5 h-3.5 text-cyan-400" />}
                  Download ZIP ({completedCount})
                </button>
              )}

              <button
                onClick={handleStartBatch}
                disabled={isBatchProcessing || items.every((it) => it.status === "completed")}
                className="flex items-center gap-1.5 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-neutral-950 text-xs font-semibold px-4 py-1.5 rounded-lg transition shadow-sm"
              >
                {isBatchProcessing ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    กำลังประมวลผล...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    เริ่มขยายทั้งหมด ({items.filter((it) => it.status !== "completed").length})
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Clean Items Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="bg-neutral-900/40 border border-neutral-800/80 hover:border-neutral-700/80 rounded-xl p-2.5 flex gap-3 relative overflow-hidden group transition"
            >
              {/* Thumbnail Container */}
              <div 
                onClick={() => (item.r2Url || item.upscaledBase64) && setPreviewModalItem(item)}
                className={`w-20 h-20 rounded-lg bg-neutral-950 shrink-0 overflow-hidden relative border border-neutral-800/80 ${
                  item.r2Url || item.upscaledBase64 ? "cursor-pointer group/thumb" : ""
                }`}
              >
                <img
                  src={item.r2Url || item.upscaledBase64 || item.previewUrl}
                  alt={item.name}
                  className="w-full h-full object-cover"
                />
                {item.status === "processing" && (
                  <div className="absolute inset-0 bg-neutral-950/70 backdrop-blur-xs flex items-center justify-center">
                    <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
                  </div>
                )}
                {(item.r2Url || item.upscaledBase64) && (
                  <div className="absolute inset-0 bg-neutral-950/40 opacity-0 group-hover/thumb:opacity-100 transition flex items-center justify-center text-white">
                    <Maximize2 className="w-4 h-4" />
                  </div>
                )}
              </div>

              {/* Info Column */}
              <div className="flex-1 flex flex-col justify-between min-w-0 py-0.5">
                <div>
                  <h4 className="text-xs font-medium text-neutral-200 truncate" title={item.name}>
                    {item.name}
                  </h4>
                  <p className="text-[11px] text-neutral-400 font-mono mt-0.5">
                    {item.originalWidth ? `${item.originalWidth}×${item.originalHeight}` : "—"}
                    {item.upscaledWidth && (
                      <span className="text-cyan-400 ml-1">
                        → {item.upscaledWidth}×{item.upscaledHeight}
                      </span>
                    )}
                  </p>
                </div>

                {/* Status & Actions */}
                <div className="flex items-center justify-between mt-2">
                  <div>
                    {item.status === "idle" && (
                      <span className="text-[11px] text-neutral-400">รอเริ่ม</span>
                    )}
                    {item.status === "processing" && (
                      <span className="text-[11px] text-cyan-400 flex items-center gap-1 font-medium">
                        รัน GPU ({item.progress}%)
                      </span>
                    )}
                    {item.status === "completed" && (
                      <span className="text-[11px] text-emerald-400 flex items-center gap-1 font-medium">
                        <CheckCircle2 className="w-3 h-3" />
                        สำเร็จ
                      </span>
                    )}
                    {item.status === "error" && (
                      <span className="text-[11px] text-rose-400 flex items-center gap-1" title={item.error}>
                        <AlertCircle className="w-3 h-3" />
                        ผิดพลาด
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-1">
                    {item.status === "completed" && (item.r2Url || item.upscaledBase64) && (
                      <button
                        onClick={() => handleDownloadSingle(item)}
                        className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition"
                        title="ดาวน์โหลดภาพนี้"
                      >
                        <Download className="w-3 h-3" />
                      </button>
                    )}
                    <button
                      onClick={() => handleRemoveItem(item.id)}
                      className="p-1 rounded hover:bg-rose-500/10 text-neutral-400 hover:text-rose-400 transition"
                      title="ลบ"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>

      {/* Lightbox / Preview Modal */}
      {previewModalItem && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl">
            <div className="h-12 border-b border-neutral-800 px-4 flex items-center justify-between">
              <span className="text-xs font-semibold text-neutral-300 truncate">
                {previewModalItem.name} ({previewModalItem.upscaledWidth}×{previewModalItem.upscaledHeight})
              </span>
              <button
                onClick={() => setPreviewModalItem(null)}
                className="p-1.5 rounded-lg hover:bg-neutral-800 text-neutral-400 hover:text-white transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-neutral-950">
              <img
                src={previewModalItem.r2Url || previewModalItem.upscaledBase64 || previewModalItem.previewUrl}
                alt={previewModalItem.name}
                className="max-h-[70vh] object-contain rounded-lg"
              />
            </div>
            <div className="h-14 border-t border-neutral-800 px-4 flex items-center justify-end">
              <button
                onClick={() => handleDownloadSingle(previewModalItem)}
                className="flex items-center gap-1.5 bg-cyan-500 hover:bg-cyan-400 text-neutral-950 text-xs font-semibold px-4 py-2 rounded-lg transition"
              >
                <Download className="w-3.5 h-3.5" />
                ดาวน์โหลดภาพนี้
              </button>
            </div>
          </div>
        </div>
      )}

      {/* BYOK Settings Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl max-w-md w-full p-6 flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                  <Key className="w-4 h-4" />
                </div>
                <h3 className="text-sm font-semibold text-white">ตั้งค่า RunPod API Key (BYOK)</h3>
              </div>
              <button
                onClick={() => setShowKeyModal(false)}
                className="p-1.5 rounded-lg hover:bg-neutral-800 text-neutral-400 hover:text-white transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="text-xs text-neutral-400 flex flex-col gap-2">
              <p>
                เว็บนี้เป็นเครื่องมือ Open-Source แบบ <strong className="text-neutral-200">Bring Your Own Key</strong> เพื่อความปลอดภัยและประหยัดค่าใช้จ่ายของคุณ
              </p>
              <div className="flex items-start gap-1.5 text-neutral-400 bg-neutral-950 p-2.5 rounded-xl border border-neutral-800">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>Key จะถูกบันทึกไว้ในเบราว์เซอร์ของคุณเท่านั้น (LocalStorage) จะไม่ถูกเก็บลงฐานข้อมูลใดๆ ทั้งสิ้น</span>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium text-neutral-300">RunPod API Key (rpa_...)</label>
              <input
                type="password"
                value={keyInputTemp}
                onChange={(e) => setKeyInputTemp(e.target.value)}
                placeholder="rpa_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-3.5 py-2 text-xs text-white placeholder:text-neutral-600 focus:outline-none focus:border-cyan-500/50 font-mono"
              />
              <p className="text-[11px] text-neutral-400">
                หาได้จาก{" "}
                <a
                  href="https://www.runpod.io/console/serverless/user/settings"
                  target="_blank"
                  rel="noreferrer"
                  className="text-cyan-400 hover:underline"
                >
                  RunPod User Settings ↗
                </a>
              </p>
            </div>

            <div className="flex items-center justify-end gap-2 mt-2">
              <button
                onClick={() => setShowKeyModal(false)}
                className="px-3 py-1.5 text-xs text-neutral-400 hover:text-white transition"
              >
                ยกเลิก
              </button>
              <button
                onClick={handleSaveApiKey}
                className="px-4 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-neutral-950 text-xs font-semibold rounded-xl transition"
              >
                บันทึก Key
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
