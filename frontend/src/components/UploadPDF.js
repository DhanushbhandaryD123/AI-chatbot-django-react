import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

function UploadPDF() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const pickFile = (selected) => {
    if (!selected) return;
    setFile(selected);
    setFileName(selected.name);
    setMessage("");
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a PDF");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setMessage("");

    try {
      const res = await fetch("http://127.0.0.1:8000/api/upload/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setMessage(data.message);
    } catch {
      setMessage("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    pickFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="upload-section">
      <h3>📄 Upload PDF</h3>

      <motion.label
        className={`dropzone ${dragActive ? "active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        whileHover={{ scale: 1.01 }}
        animate={{
          borderColor: dragActive ? "#60a5fa" : "rgba(255,255,255,0.25)",
          scale: dragActive ? 1.02 : 1,
        }}
      >
        <motion.div
          className="dropzone-icon"
          animate={{ y: dragActive ? -6 : 0, rotate: dragActive ? -8 : 0 }}
          transition={{ type: "spring", stiffness: 250, damping: 12 }}
        >
          📁
        </motion.div>

        <span>{fileName || "Drag & drop a PDF here, or click to browse"}</span>

        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => pickFile(e.target.files[0])}
          hidden
        />
      </motion.label>

      <motion.button
        onClick={handleUpload}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        disabled={uploading}
      >
        {uploading ? (
          <motion.span
            animate={{ rotate: 360 }}
            transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
            style={{ display: "inline-block" }}
          >
            ⏳
          </motion.span>
        ) : (
          "⬆ Upload"
        )}
      </motion.button>

      <AnimatePresence>
        {message && (
          <motion.p
            className="upload-message"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            {message}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}

export default UploadPDF;
