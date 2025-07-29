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
      className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-8 cursor-pointer
        ${isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300"}`}
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
