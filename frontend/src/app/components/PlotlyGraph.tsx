"use client";

import React from 'react';
import Plot from 'react-plotly.js';

const PlotlyGraph = () => {
  const data = [
    {
      x: [1, 2, 3],
      y: [4, 5, 6],
      z: [7, 8, 9],
      mode: 'markers',
      marker: { size: 12 },
      type: 'scatter3d',
    },
  ];

  const layout = {
    width: 1000,
    height: 400,
    title: '3D Scatter Plot',
    scene: {
      xaxis: { title: 'X Axis' },
      yaxis: { title: 'Y Axis' },
      zaxis: { title: 'Z Axis', range: [6, 10], autorange: false },
    },
  };

  return <Plot data={data} layout={layout} />;
};

export default PlotlyGraph;