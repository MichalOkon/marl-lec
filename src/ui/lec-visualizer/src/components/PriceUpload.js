import React, {useState} from 'react';
import Papa from 'papaparse';
import './DragDrop.css';

function PriceUploader({onImportPriceParsed, onExportPriceParsed}) {
    const [importDragActive, setImportDragActive] = useState(false);
    const [exportDragActive, setExportDragActive] = useState(false);
    const [importUploaded, setImportUploaded] = useState(false);
    const [exportUploaded, setExportUploaded] = useState(false);

    const handleFile = (file, callback, markUploaded) => {
        if (file) {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    const data = results.data;
                    const prices = data.map((row) => row.price || Object.values(row)[0]);
                    callback(prices);
                    markUploaded(true);
                }
            });
        }
    };

    const handleImportDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setImportDragActive(false);
        const file = e.dataTransfer.files[0];
        handleFile(file, onImportPriceParsed, setImportUploaded);
    };

    const handleExportDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setExportDragActive(false);
        const file = e.dataTransfer.files[0];
        handleFile(file, onExportPriceParsed, setExportUploaded);
    };

    return (
        <div>
            <div style={{display: 'flex', width: '100%', justifyContent: 'space-between'}}>
                <div
                    className={`left-zone drag-drop-zone${importDragActive ? ' active' : ''}${
                        importUploaded ? ' uploaded' : ''
                    } drag-drop-zone-small`}
                    onDragOver={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setImportDragActive(true);
                    }}
                    onDragLeave={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setImportDragActive(false);
                    }}
                    onDrop={handleImportDrop}
                >
                    {importUploaded && <span className="check-mark">✓</span>}
                    {importDragActive
                        ? "Release to upload Import CSV"
                        : "Drop Import Price CSV"}
                </div>
                <div
                    className={`drag-drop-zone${exportDragActive ? ' active' : ''}${
                        exportUploaded ? ' uploaded' : ''
                    } drag-drop-zone-small`}
                    onDragOver={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setExportDragActive(true);
                    }}
                    onDragLeave={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setExportDragActive(false);
                    }}
                    onDrop={handleExportDrop}
                >
                    {exportUploaded && <span className="check-mark">✓</span>}
                    {exportDragActive
                        ? "Release to upload Export CSV"
                        : "Drop Export Price CSV"}
                </div>
            </div>
        </div>
    );
}

export default PriceUploader;
