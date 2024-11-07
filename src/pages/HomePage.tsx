import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../features/auth/context/AuthContext';
import { 
  Brain, 
  BarChart2, 
  TrendingUp, 
  Package, 
  Factory,
  Network,
  CheckCircle,
  Truck,
  Users
} from 'lucide-react';

const supplyChainModules = [
  {
    icon: <TrendingUp className="w-16 h-16 text-primary" />,
    title: "Demand Planning",
    description: "AI-powered demand forecasting and planning optimization with collaborative insights",
    tagline: "Predict. Plan. Perform.",
    features: ["Advanced forecasting algorithms", "Real-time market analysis", "Collaborative planning"],
    path: "/modules/demand-planning"
  },
  {
    icon: <Factory className="w-16 h-16 text-primary" />,
    title: "Factory Planning",
    description: "Intelligent production scheduling and resource allocation for maximum efficiency",
    tagline: "Optimize. Execute. Excel.",
    features: ["Smart resource allocation", "Production optimization", "Capacity planning"],
    path: "/modules/factory"
  },
  {
    icon: <Network className="w-16 h-16 text-primary" />,
    title: "Enterprise Supply Planning",
    description: "End-to-end supply chain optimization and strategic coordination",
    tagline: "Connect. Coordinate. Control.",
    features: ["Network optimization", "Strategic planning", "Risk management"],
    path: "/modules/enterprise"
  },
  {
    icon: <Truck className="w-16 h-16 text-primary" />,
    title: "Order Promising",
    description: "Real-time order management and delivery optimization system",
    tagline: "Promise. Deliver. Satisfy.",
    features: ["Real-time ATP/CTP", "Dynamic routing", "Delivery optimization"],
    path: "/modules/order-promising"
  },
  {
    icon: <Package className="w-16 h-16 text-primary" />,
    title: "Inventory Liquidation",
    description: "Smart inventory management and excess stock optimization",
    tagline: "Analyze. Optimize. Liquidate.",
    features: ["Stock optimization", "Markdown planning", "Channel allocation"],
    path: "/modules/inventory"
  },
  {
    icon: <Users className="w-16 h-16 text-primary" />,
    title: "Collaborative Forecast Optimization",
    description: "Multi-stakeholder forecasting and planning platform",
    tagline: "Collaborate. Align. Succeed.",
    features: ["Stakeholder alignment", "Consensus planning", "Performance tracking"],
    path: "/modules/collaborative"
  }
];

export default function HomePage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const modulesRef = useRef<HTMLDivElement>(null);

  const handleStartAnalysis = () => {
    modulesRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleModuleClick = (path: string) => {
    if (!user) {
      navigate('/login', { state: { returnPath: path } });
    } else {
      navigate('/chat');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-secondary-dark to-secondary pt-16">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 lg:mt-16 lg:px-8 xl:mt-28">
              <div className="text-center hero-animate">
                <div className="flex justify-center mb-8">
                  <Brain className="w-24 h-24 text-primary animate-pulse" />
                </div>
                <h1 className="text-4xl tracking-tight font-extrabold text-white sm:text-5xl md:text-6xl">
                  <span className="block">Next-Generation</span>
                  <span className="block gradient-text">Supply Chain Intelligence</span>
                </h1>
                <p className="mt-3 max-w-md mx-auto text-base text-gray-300 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
                  Transform your supply chain with advanced AI analytics and Root Cause Analysis. 
                  Make data-driven decisions faster and smarter.
                </p>
                <div className="mt-5 max-w-md mx-auto sm:flex sm:justify-center md:mt-8">
                  <div className="rounded-md shadow">
                    <button
                      onClick={handleStartAnalysis}
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-secondary-dark bg-primary hover:bg-primary-dark transition-all duration-300 transform hover:scale-105 md:py-4 md:text-lg md:px-10 hover-effect"
                    >
                      Start Analysis
                    </button>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Modules Section */}
      <div ref={modulesRef} className="py-12 bg-secondary-dark/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">
              Supply Chain Modules
            </h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-white sm:text-4xl">
              Comprehensive Supply Chain Solutions
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-300 lg:mx-auto">
              Select a module to begin your AI-powered supply chain optimization journey
            </p>
          </div>

          <div className="mt-16">
            <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
              {supplyChainModules.map((module, index) => (
                <div
                  key={index}
                  onClick={() => handleModuleClick(module.path)}
                  className="relative group bg-secondary p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-primary rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer module-card"
                >
                  <div>
                    <div className="flex items-center justify-center h-24 w-24 rounded-md bg-secondary-dark text-white mx-auto feature-icon">
                      {module.icon}
                    </div>
                    <div className="mt-8">
                      <h3 className="text-lg font-medium text-white text-center">
                        {module.title}
                      </h3>
                      <p className="mt-2 text-sm text-primary text-center font-medium">
                        {module.tagline}
                      </p>
                      <p className="mt-2 text-base text-gray-300 text-center">
                        {module.description}
                      </p>
                      <ul className="mt-4 space-y-2">
                        {module.features.map((feature, idx) => (
                          <li key={idx} className="flex items-center text-gray-300">
                            <CheckCircle className="h-4 w-4 text-primary mr-2" />
                            <span className="text-sm">{feature}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  <div className="absolute bottom-6 left-0 right-0 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <span className="inline-flex items-center px-4 py-2 text-sm font-medium text-primary">
                      Start Analysis →
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-secondary">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-primary font-semibold tracking-wide uppercase">
              Root Cause Analysis
            </h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-white sm:text-4xl">
              AI-Powered Problem Solving
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-300 lg:mx-auto">
              Identify and resolve supply chain issues faster with our advanced AI analytics
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-secondary-dark">
                  <Brain className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-white">
                  Intelligent Analysis
                </p>
                <p className="mt-2 ml-16 text-base text-gray-300">
                  Advanced pattern recognition to identify root causes quickly
                </p>
              </div>

              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-secondary-dark">
                  <BarChart2 className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-white">
                  Data-Driven Solutions
                </p>
                <p className="mt-2 ml-16 text-base text-gray-300">
                  Get actionable recommendations based on historical data and AI insights
                </p>
              </div>

              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-secondary-dark">
                  <Network className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-white">
                  End-to-End Visibility
                </p>
                <p className="mt-2 ml-16 text-base text-gray-300">
                  Complete transparency across your entire supply chain network
                </p>
              </div>

              <div className="relative">
                <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-primary text-secondary-dark">
                  <CheckCircle className="h-6 w-6" />
                </div>
                <p className="ml-16 text-lg leading-6 font-medium text-white">
                  Predictive Quality
                </p>
                <p className="mt-2 ml-16 text-base text-gray-300">
                  Anticipate and prevent quality issues before they occur
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}