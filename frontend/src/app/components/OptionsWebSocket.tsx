"use client";

import React, { useEffect, useRef, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale } from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale);

const OptionsWebSocket = () => {
  const [bidPrices, setBidPrices] = useState<number[]>([]);
  const [askPrices, setAskPrices] = useState<number[]>([]);
  const [timestamps, setTimestamps] = useState<string[]>([]);
  const chartRef = useRef<any>(null);
  const apiKey = process.env.REACT_APP_API_KEY;
  const contractSymbol = 'NKE241129C00076000'; // Replace with your desired options contract symbol

  useEffect(() => {
    const client = new WebSocket('wss://socket.polygon.io/options'); // Polygon.io Options WebSocket URL

    client.onopen = () => {
      console.log('WebSocket Client Connected');
      client.send(JSON.stringify({ action: 'auth', params: apiKey })); // Use the apiKey variable
      client.send(JSON.stringify({ action: 'subscribe', params: `Q.O:${contractSymbol}` })); // Subscribe to the desired options contract
    };

    client.onmessage = (message) => {
      const data = JSON.parse(message.data);
      console.log('Received WebSocket data:', data);

      if (data.ev === 'Q') { // Check for options trade events
        const { bp, ap, t } = data; // `bp` = bid price, `ap` = ask price, `t` = timestamp
        const time = new Date(t).toLocaleTimeString();

        setBidPrices((prev) => {
          const updated = [...prev, bp];
          return updated.length > 150 ? updated.slice(1) : updated; // Limit to 150 points
        });

        setAskPrices((prev) => {
          const updated = [...prev, ap];
          return updated.length > 150 ? updated.slice(1) : updated; // Limit to 150 points
        });

        setTimestamps((prev) => {
          const updated = [...prev, time];
          return updated.length > 150 ? updated.slice(1) : updated; // Limit timestamps to match data points
        });
      } else {
        console.warn('Invalid WebSocket data:', data);
      }
    };

    client.onerror = (error) => {
      console.error('WebSocket Error:', error);
      console.log('WebSocket Ready State:', client.readyState);

    };

    client.onclose = (event) => {
      console.log('WebSocket Client Disconnected');
      console.warn('WebSocket Closed:', event);
      console.log('Close Code:', event.code);
      console.log('Close Reason:', event.reason);
    };

    return () => {
      client.close();
    };
  }, [contractSymbol]); // Add contractSymbol as a dependency

  const chartData = {
    labels: timestamps,
    datasets: [
      {
        label: 'Bid Price',
        data: bidPrices,
        borderColor: 'blue',
        fill: false,
      },
      {
        label: 'Ask Price',
        data: askPrices,
        borderColor: 'orange',
        fill: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: (context) => `Price: $${context.raw}`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Price',
        },
      },
    },
  };

  return (
    <div>
      <h2>Real-Time Options Bid and Ask Prices ({contractSymbol})</h2>
      <Line data={chartData} options={chartOptions} ref={chartRef} />
    </div>
  );
};

export default OptionsWebSocket;
