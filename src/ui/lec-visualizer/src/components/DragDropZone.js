import React, { useState } from 'react';
import Papa from 'papaparse';
import './DragDrop.css';

function DragDropZone({ onFileParsed, onLoadingChange }) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploaded, setUploaded] = useState(false);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      if (onLoadingChange) onLoadingChange(true);
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          onFileParsed(results.data);
          if (onLoadingChange) onLoadingChange(false);
          setUploaded(true);
        }
      });
    }
  };

  const zoneClasses = `drag-drop-zone${isDragging ? ' active' : ''}${
    uploaded ? ' uploaded' : ''
  }`;

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      className={zoneClasses}
    >
      {uploaded && <span className="check-mark">✓</span>}
      {isDragging ? 'Release to upload file' : 'Drop Energy Data CSV File'}
    </div>
  );
}

export default DragDropZone;
