import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';

function EnergyCharts({ logs, importPrices, exportPrices, range }) {
  const slicedData = useMemo(() => {
    const sliceData = (dataArray) => dataArray.slice(range[0], range[1]);
    let result = {};
    Object.keys(logs).forEach((key) => {
      result[key] = sliceData(logs[key]);
    });
    return result;
  }, [logs, range]);

  const slicedImportPrices = useMemo(() => {
    return importPrices ? importPrices.slice(range[0], range[1]) : null;
  }, [importPrices, range]);

  const slicedExportPrices = useMemo(() => {
    return exportPrices ? exportPrices.slice(range[0], range[1]) : null;
  }, [exportPrices, range]);

  const buildPlot = (title, energyKeys) => {
    const timesteps = slicedData[energyKeys[0]].map((_, i) => i);

    const traces = energyKeys.map((key) => ({
      x: timesteps,
      y: slicedData[key],
      type: 'bar',
      name: key,
    }));

    if (slicedImportPrices) {
      traces.push({
        x: timesteps,
        y: slicedImportPrices,
        type: 'scatter',
        mode: 'lines',
        name: 'Import Prices',
        line: { color: 'red', dash: 'dash' },
        yaxis: 'y2'
      });
    }

    if (slicedExportPrices) {
      traces.push({
        x: timesteps,
        y: slicedExportPrices,
        type: 'scatter',
        mode: 'lines',
        name: 'Export Prices',
        line: { color: 'green', dash: 'dash' },
        yaxis: 'y2'
      });
    }

    const layout = {
      title,
      barmode: 'stack',
      xaxis: { title: 'Timesteps' },
      yaxis: { title: 'Energy Values [kWh]' },
      yaxis2: {
        title: 'Price',
        overlaying: 'y',
        side: 'right',
        showgrid: false,
      },
      hovermode: 'closest',
    };

    return <Plot data={traces} layout={layout} style={{ width: '100%', height: '500px' }} />;
  };

  return (
    <div>
      {buildPlot('Incoming Energy', [
        'average_imported_energy',
        'average_discharge',
        'common_battery_discharge',
        'average_locally_imported_energy',
        'average_produced_energy',
      ])}
      {buildPlot('Outgoing Energy', [
        'average_exported_energy',
        'average_charge',
        'common_battery_charge',
        'average_locally_exported_energy',
        'average_load',
      ])}
    </div>
  );
}

export default EnergyCharts;
