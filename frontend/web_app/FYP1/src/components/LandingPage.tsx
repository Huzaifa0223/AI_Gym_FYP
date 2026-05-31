import { Dumbbell, Target, Brain, TrendingUp, CheckCircle, ArrowRight, Users, Award, Zap } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 w-full bg-slate-900/80 backdrop-blur-lg border-b border-white/10">
        <div className="w-full px-6 md:px-12 lg:px-20 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
                <Dumbbell className="w-6 h-6 text-white" />
              </div>
              <span className="text-white text-xl">AI Gym Trainer</span>
            </div>
            
            <div className="hidden md:flex items-center gap-8">
              <a href="#home" className="text-gray-300 hover:text-white transition-colors">Home</a>
              <a href="#about" className="text-gray-300 hover:text-white transition-colors">About Us</a>
              <a href="#features" className="text-gray-300 hover:text-white transition-colors">Features</a>
            </div>

            <button
              onClick={onGetStarted}
              className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-6 py-2 rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-blue-500/50"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section - Home */}
      <section id="home" className="pt-24 pb-20 px-6 md:px-12 lg:px-20 w-full">
        <div className="w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="text-center lg:text-left">
              <div className="inline-flex items-center gap-2 bg-blue-500/20 border border-blue-500/30 rounded-full px-4 py-2 mb-6">
                <Zap className="w-4 h-4 text-blue-400" />
                <span className="text-blue-400">AI-Powered Fitness Coach</span>
              </div>
              
              <h1 className="text-white text-5xl md:text-6xl mb-6">
                Transform Your Body with{' '}
                <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  AI Precision
                </span>
              </h1>
              
              <p className="text-gray-400 text-lg mb-8 max-w-xl">
                Get real-time form corrections, personalized workout plans, and achieve your fitness goals faster with our advanced AI computer vision technology.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                <button
                  onClick={onGetStarted}
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-8 py-4 rounded-xl hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-blue-500/50 flex items-center justify-center gap-2"
                >
                  Start Your Journey
                  <ArrowRight className="w-5 h-5" />
                </button>
                
                <a
                  href="#about"
                  className="bg-white/10 backdrop-blur-lg border border-white/20 text-white px-8 py-4 rounded-xl hover:bg-white/20 transition-all flex items-center justify-center gap-2"
                >
                  Learn More
                </a>
              </div>

              <div className="flex items-center gap-8 mt-12 justify-center lg:justify-start">
                <div>
                  <p className="text-white text-3xl mb-1">10K+</p>
                  <p className="text-gray-400">Active Users</p>
                </div>
                <div className="w-px h-12 bg-white/20" />
                <div>
                  <p className="text-white text-3xl mb-1">95%</p>
                  <p className="text-gray-400">Accuracy Rate</p>
                </div>
                <div className="w-px h-12 bg-white/20" />
                <div>
                  <p className="text-white text-3xl mb-1">1M+</p>
                  <p className="text-gray-400">Workouts Tracked</p>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-3xl blur-3xl" />
              <div className="relative bg-gradient-to-br from-blue-500/10 to-cyan-500/10 rounded-3xl p-2 border border-blue-500/30">
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1584827386916-b5351d3ba34b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmaXRuZXNzJTIwZ3ltJTIwd29ya291dHxlbnwxfHx8fDE3NjM2NDkwNzF8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                  alt="Person doing gym workout"
                  className="w-full h-auto rounded-2xl animate-float"
                />
                
                {/* Floating Stats Cards */}
                <div className="absolute top-8 -left-4 bg-slate-900/90 backdrop-blur-lg rounded-xl p-4 border border-blue-500/30 shadow-xl animate-float-delay-1">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                      <CheckCircle className="w-5 h-5 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-white">Form Correct!</p>
                      <p className="text-blue-400">95% Accuracy</p>
                    </div>
                  </div>
                </div>

                <div className="absolute bottom-8 -right-4 bg-slate-900/90 backdrop-blur-lg rounded-xl p-4 border border-cyan-500/30 shadow-xl animate-float-delay-2">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                      <TrendingUp className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <p className="text-white">12 Reps</p>
                      <p className="text-cyan-400">Great Progress!</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 md:px-12 lg:px-20 w-full">
        <div className="w-full">
          <div className="text-center mb-16">
            <h2 className="text-white text-4xl mb-4">Why Choose AI Gym Trainer?</h2>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Experience the future of fitness with cutting-edge AI technology designed to help you achieve your goals
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Brain className="w-7 h-7 text-blue-400" />
              </div>
              <h3 className="text-white text-xl mb-3">AI-Powered Analysis</h3>
              <p className="text-gray-400">
                Advanced computer vision technology analyzes your form in real-time, ensuring every rep counts
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Target className="w-7 h-7 text-cyan-400" />
              </div>
              <h3 className="text-white text-xl mb-3">Personalized Plans</h3>
              <p className="text-gray-400">
                Get workout recommendations tailored to your age, goals, and fitness level
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <TrendingUp className="w-7 h-7 text-blue-400" />
              </div>
              <h3 className="text-white text-xl mb-3">Real-time Feedback</h3>
              <p className="text-gray-400">
                Instant corrections and guidance help you maintain perfect form throughout your workout
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Award className="w-7 h-7 text-cyan-400" />
              </div>
              <h3 className="text-white text-xl mb-3">Track Progress</h3>
              <p className="text-gray-400">
                Monitor your improvements with detailed analytics and workout history
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Users className="w-7 h-7 text-blue-400" />
              </div>
              <h3 className="text-white text-xl mb-3">All Age Groups</h3>
              <p className="text-gray-400">
                Suitable for everyone from teens (10-16) to adults (17-45) and seniors (46+)
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
              <div className="w-14 h-14 bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Dumbbell className="w-7 h-7 text-cyan-400" />
              </div>
              <h3 className="text-white text-xl mb-3">Multiple Goals</h3>
              <p className="text-gray-400">
                Whether you want to gain weight, lose weight, or maintain - we've got you covered
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* About Us Section */}
      <section id="about" className="py-20 px-6 md:px-12 lg:px-20 w-full bg-white/5">
        <div className="w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-white text-4xl mb-6">About AI Gym Trainer</h2>
              <p className="text-gray-400 text-lg mb-6">
                We're revolutionizing the fitness industry by combining artificial intelligence with expert fitness knowledge. Our mission is to make professional-grade fitness coaching accessible to everyone, anywhere.
              </p>
              
              <div className="space-y-4 mb-8">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="text-white mb-1">Advanced Computer Vision</h4>
                    <p className="text-gray-400">Our AI uses state-of-the-art computer vision to detect and analyze your movements with 95% accuracy</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="text-white mb-1">Expert-Designed Workouts</h4>
                    <p className="text-gray-400">Every exercise recommendation is based on proven fitness science and best practices</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="text-white mb-1">Privacy First</h4>
                    <p className="text-gray-400">Your workout data is processed securely, and your privacy is our top priority</p>
                  </div>
                </div>
              </div>

              <button
                onClick={onGetStarted}
                className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-8 py-4 rounded-xl hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-blue-500/50 flex items-center gap-2"
              >
                Start Training Now
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-lg rounded-2xl p-6 border border-blue-500/30">
                <p className="text-blue-400 text-4xl mb-2">10K+</p>
                <p className="text-gray-300">Happy Users</p>
              </div>
              
              <div className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 backdrop-blur-lg rounded-2xl p-6 border border-cyan-500/30">
                <p className="text-cyan-400 text-4xl mb-2">1M+</p>
                <p className="text-gray-300">Workouts</p>
              </div>
              
              <div className="bg-gradient-to-br from-cyan-500/20 to-blue-600/20 backdrop-blur-lg rounded-2xl p-6 border border-cyan-500/30">
                <p className="text-cyan-400 text-4xl mb-2">95%</p>
                <p className="text-gray-300">Accuracy</p>
              </div>
              
              <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-lg rounded-2xl p-6 border border-blue-500/30">
                <p className="text-blue-400 text-4xl mb-2">24/7</p>
                <p className="text-gray-300">Available</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 md:px-12 lg:px-20 w-full">
        <div className="w-full">
          <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-3xl p-12 text-center">
            <h2 className="text-white text-4xl mb-4">Ready to Transform Your Fitness Journey?</h2>
            <p className="text-blue-100 text-lg mb-8">
              Join thousands of users who are already achieving their fitness goals with AI Gym Trainer
            </p>
            <button
              onClick={onGetStarted}
              className="bg-white text-blue-600 px-8 py-4 rounded-xl hover:bg-gray-100 transition-all shadow-lg"
            >
              Get Started Free
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 md:px-12 lg:px-20 w-full border-t border-white/10">
        <div className="w-full text-center text-gray-400">
          <p>© 2025 AI Gym Trainer. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
