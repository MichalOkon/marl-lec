import React, { useState } from 'react';
import DragDropZone from './components/DragDropZone';
import PriceUploader from './components/PriceUpload';
import EnergyCharts from './components/EnergyCharts';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import useDebounce from './hooks/useDebounce';
import './App.css';

function App() {
  const [logs, setLogs] = useState(null);
  const [importPrices, setImportPrices] = useState(null);
  const [exportPrices, setExportPrices] = useState(null);
  const [range, setRange] = useState([0, 500]);
  const [loading, setLoading] = useState(false);

  const debouncedRange = useDebounce(range, 500);

  const handleFileParsed = (data) => {
    let newLogs = {};
    Object.keys(data[0]).forEach((key) => {
      newLogs[key] = data.map((row) => row[key]);
    });
    setLogs(newLogs);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Energy Data Dashboard</h1>
      </header>
      <DragDropZone onFileParsed={handleFileParsed} onLoadingChange={setLoading} />
      {loading && (
        <div style={{ textAlign: 'center', margin: '20px' }}>
          <div className="spinner">Loading...</div>
        </div>
      )}
      <PriceUploader
        onImportPriceParsed={setImportPrices}
        onExportPriceParsed={setExportPrices}
      />
      {logs && (
        <div className="slider-container">
          <h4>Select Data Range</h4>
          <Slider
            range
            min={0}
            max={logs['average_imported_energy'].length}
            value={range}
            onChange={setRange}
            allowCross={false}
            trackStyle={[{ backgroundColor: '#2c3e50' }]}
            handleStyle={[
              { borderColor: '#2c3e50' },
              { borderColor: '#2c3e50' },
            ]}
          />
          <div style={{ marginTop: '10px' }}>
            Showing indices: {range[0]} to {range[1]}
          </div>
        </div>
      )}
      {logs && (
        <EnergyCharts
          logs={logs}
          importPrices={importPrices}
          exportPrices={exportPrices}
          range={debouncedRange}
        />
      )}
    </div>
  );
}

export default App;
