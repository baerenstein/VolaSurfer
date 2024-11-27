"use client"

import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale } from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale);

const RidgePlot = ({ data }) => {
  return (
    <div>
      {data.map((dataset, index) => (
        <div key={index} style={{ position: 'relative', height: '100px' }}>
          <Line
            data={{
              labels: dataset.labels,
              datasets: [
                {
                  label: dataset.label,
                  data: dataset.values,
                  borderColor: `rgba(75, 192, 192, ${1 - index * 0.2})`, // Adjust opacity for layering
                  backgroundColor: `rgba(75, 192, 192, ${0.2 + index * 0.1})`,
                  fill: true,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: 'top', // Move legend to the top
                  align: 'end', // Align legend to the right
                },
              },
              scales: {
                x: {
                  display: false, // Hide x-axis
                },
                y: {
                  display: false, // Hide y-axis
                },
              },
            }}
          />
        </div>
      ))}
    </div>
  );
};

export default RidgePlot; 