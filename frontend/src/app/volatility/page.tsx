import Header from '../components/Header';
import Sidebar from '../components/Sidebar';
import LineChart from '../components/LineChart';
import RidgePlot from '../components/RidgePlot';

const generateVolatilityData = () => {
  const labels = Array.from({ length: 10 }, (_, i) => `Day ${i + 1}`);
  const values = Array.from({ length: 10 }, () => Math.random() * 50); // Synthetic volatility data
  return { labels, values, label: 'Synthetic Volatility Data' };
};

const generateRidgeData = () => {
  return Array.from({ length: 5 }, (_, i) => {
    const values = Array.from({ length: 10 }, () => Math.random() * (50 - i * 5) + i * 5); // Varying ranges
    return {
      labels: Array.from({ length: 10 }, (_, j) => `Day ${j + 1}`),
      values,
      label: `Distribution ${i + 1}`,
    };
  });
};

export default function VolatilityAnalysis() {
  const ridgeData = generateRidgeData();

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
          <div className="container mx-auto px-6 py-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Volatility Analysis</h2>
            <div className="bg-white shadow rounded-lg p-4">
              <RidgePlot data={ridgeData} />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}