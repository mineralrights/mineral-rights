"use client";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

type Props = {
  onSelect: (files: File[]) => void;
};

export default function PDFUpload({ onSelect }: Props) {
  const onDrop = useCallback((accepted: File[]) => {
    onSelect(accepted);
  }, [onSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "application/pdf": [".pdf"] },
    onDrop
  });

  return (
    <div
      {...getRootProps()}
      className={`flex flex-col items-center justify-center border-2 rounded-lg
        p-8 cursor-pointer transition-colors
        ${isDragActive
          ? "border-[color:var(--accent)] bg-emerald-50/40"
          : "border-gray-300 hover:border-[color:var(--accent)]"}`}
    >
      <input {...getInputProps()} multiple />
      <p className="text-gray-600">
        {isDragActive
          ? "Drop the PDFs here…"
          : "Drag & drop PDFs here or click to browse"}
      </p>
    </div>
  );
}
