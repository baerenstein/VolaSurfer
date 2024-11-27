import Header from '../components/Header';
import Sidebar from '../components/Sidebar';

const services = [
  {
    category: "Algorithmic Trading Development",
    description: "Design and implementation of custom trading algorithms with focus on market microstructure and signal generation",
    deliverables: "Trading algorithm with documentation, execution framework, monitoring tools",
  },
  {
    category: "Hedging Strategy Development",
    description: "Creation of dynamic hedging solutions with portfolio risk analysis and instrument selection",
    deliverables: "Hedging strategy document, implementation guide, risk metrics",
  },
  {
    category: "Machine Learning & Financial Modeling",
    description: "Development of predictive models using ML techniques for market forecasting",
    deliverables: "ML models, prediction framework, performance analytics",
  },
  {
    category: "Risk Management & Validation",
    description: "Comprehensive risk assessment including stress testing and regulatory compliance",
    deliverables: "Risk validation report, compliance documentation, control framework",
  },
  {
    category: "Backtesting & Optimization",
    description: "Historical performance analysis and parameter optimization for trading strategies",
    deliverables: "Backtesting report, optimized parameters, robustness metrics",
  },
  {
    category: "Data Engineering",
    description: "Development of data pipelines and real-time processing systems",
    deliverables: "Data infrastructure setup, pipeline documentation, monitoring system",
  },
  {
    category: "Consultation & Training",
    description: "Knowledge transfer sessions and hands-on training for strategy implementation",
    deliverables: "Training materials, best practices guide, code review documentation",
  },
];

export default function Services() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-100">
          <div className="container mx-auto px-6 py-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Services</h2>
            <div className="bg-white shadow rounded-lg p-4">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Service Category</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Key Deliverables</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {services.map((service, index) => (
                    <tr key={index}>
                      <td className="px-4 py-4 whitespace-normal text-sm text-gray-900">{service.category}</td>
                      <td className="px-4 py-4 whitespace-normal text-sm text-gray-900">{service.description}</td>
                      <td className="px-4 py-4 whitespace-normal text-sm text-gray-900">{service.deliverables}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
} 