import Link from 'next/link'

export default function Sidebar() {
  return (
    <nav className="bg-white shadow-lg w-64 h-screen">
      <div className="px-4 py-6">
        <ul className="space-y-2">
          <li>
            <Link href="/" className="block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded">
              Overview
            </Link>
          </li>
          <li>
            <Link href="/options" className="block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded">
              Options Data
            </Link>
          </li>
          <li>
            <Link href="/volatility" className="block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded">
              Volatility Analysis
            </Link>
          </li>
          <li>
            <Link href="/graph3d" className="block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded">
              3D Graph
            </Link>
          </li>
          <li>
            <Link href="/services" className="block px-4 py-2 text-gray-700 hover:bg-gray-200 rounded">
              Services
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  )
}