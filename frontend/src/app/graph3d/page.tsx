import Header from '../components/Header';
import Sidebar from '../components/Sidebar';
import PlotlyGraph from '../components/PlotlyGraph';

export default function Graph3DPage() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
          <div className="container mx-auto px-6 py-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Plotly Graph</h2>
            <PlotlyGraph />
          </div>
        </main>
      </div>
    </div>
  );
} 