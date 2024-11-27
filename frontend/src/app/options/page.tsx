"use client"
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import { useEffect, useState } from 'react';
import axios from 'axios';

interface VolatilityData {
  insights: string[]; // Adjust this type based on the actual structure of your data
}

export default function Home() {
  const [volatilityData, setVolatilityData] = useState<VolatilityData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/volatility-data'); // Ensure this endpoint is correct
        setVolatilityData(response.data);
      } catch (error) {
        console.error("Error fetching volatility data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
          <div className="container mx-auto px-6 py-8">
            <h2 className="text-2xl font-semibold text-gray-900">VolaSurfer Dashboard</h2>
            <p className="text-gray-700 mt-2">
              Empowering retail traders to play like institutions by providing insights into options and volatility analysis.
            </p>
            <div className="bg-white shadow-md rounded-lg p-4 mt-4 h-96">
              <h3 className="text-xl font-semibold text-gray-800">Key Insights</h3>
              <p className="text-gray-700">Here are some key insights based on the latest volatility data:</p>
              <ul className="list-disc list-inside mt-2">
                {volatilityData && volatilityData.insights.map((insight: string, index: number) => (
                  <li key={index} className="text-gray-600">{insight}</li>
                ))}
              </ul>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}