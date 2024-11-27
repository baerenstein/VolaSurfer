import Header from './components/Header'
import Sidebar from './components/Sidebar'

export default function Home() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
          <div className="container mx-auto px-6 py-8">
            <h2 className="text-2xl font-semibold text-gray-900">Dashboard Overview</h2>
            <div className="bg-white shadow-md rounded-lg p-4 mt-4 h-96">
              <p className="text-gray-700">This is a box inside the overview.</p>
            </div>
            {/* Add dashboard content here */}
          </div>
        </main>
      </div>
    </div>
  )
}
