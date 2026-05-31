import { useState } from 'react';
import { Dumbbell, Mail, Lock, User, Calendar, Weight, Ruler, AlertCircle } from 'lucide-react';
import { authAPI, setToken, type User as UserType } from '../services/api';

interface LoginSignupProps {
  onLogin: (user: UserType) => void;
}

export function LoginSignup({ onLogin }: LoginSignupProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    age: '',
    weight: '',
    height: '',
    goal: 'gain' as 'gain' | 'loss' | 'maintenance',
    experienceMonths: '0',
    gender: 'male' as 'male' | 'female' | 'other',
    gymTiming: 'morning'
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      if (isLogin) {
        // Login
        const response = await authAPI.login({
          email: formData.email,
          password: formData.password
        });

        setToken(response.token);
        onLogin(response.user);
      } else {
        // Signup
        const response = await authAPI.signup({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          age: parseInt(formData.age) || 25,
          weight: parseInt(formData.weight) || 70,
          height: parseInt(formData.height) || 175,
          goal: formData.goal,
          experienceMonths: parseInt(formData.experienceMonths) || 0,
          gender: formData.gender,
          gymTiming: formData.gymTiming
        });

        setToken(response.token);
        onLogin(response.user);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo & Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl mb-4">
            <Dumbbell className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-white text-4xl mb-2">AI Gym Trainer</h1>
          <p className="text-gray-400">Your personal AI-powered fitness coach</p>
        </div>

        {/* Form Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl border border-white/20">
          {/* Toggle Buttons */}
          <div className="flex gap-2 mb-6 bg-black/20 rounded-lg p-1">
            <button
              onClick={() => { setIsLogin(true); setError(null); }}
              className={`flex-1 py-2 rounded-md transition-all ${isLogin
                ? 'bg-blue-500 text-white'
                : 'text-gray-300 hover:text-white'
                }`}
            >
              Login
            </button>
            <button
              onClick={() => { setIsLogin(false); setError(null); }}
              className={`flex-1 py-2 rounded-md transition-all ${!isLogin
                ? 'bg-blue-500 text-white'
                : 'text-gray-300 hover:text-white'
                }`}
            >
              Sign Up
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label className="block text-gray-300 mb-2">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                    placeholder="Enter your name"
                    required={!isLogin}
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-gray-300 mb-2">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                  placeholder="Enter your email"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-gray-300 mb-2">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                  placeholder="Enter your password"
                  required
                />
              </div>
            </div>

            {!isLogin && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-gray-300 mb-2">Age</label>
                    <div className="relative">
                      <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="number"
                        value={formData.age}
                        onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                        className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                        placeholder="Age"
                        required={!isLogin}
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-gray-300 mb-2">Weight (kg)</label>
                    <div className="relative">
                      <Weight className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="number"
                        value={formData.weight}
                        onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
                        className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                        placeholder="kg"
                        required={!isLogin}
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 mb-2">Height (cm)</label>
                  <div className="relative">
                    <Ruler className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                    <input
                      type="number"
                      value={formData.height}
                      onChange={(e) => setFormData({ ...formData, height: e.target.value })}
                      className="w-full bg-black/30 border border-white/20 rounded-lg px-10 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
                      placeholder="Enter height in cm"
                      required={!isLogin}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-gray-300 mb-2">Fitness Goal</label>
                  <select
                    value={formData.goal}
                    onChange={(e) => setFormData({ ...formData, goal: e.target.value as any })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    required={!isLogin}
                  >
                    <option value="gain">Gain Weight</option>
                    <option value="loss">Loss Weight</option>
                    <option value="maintenance">Maintenance</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-300 mb-2">Gym Experience (months)</label>
                  <select
                    value={formData.experienceMonths}
                    onChange={(e) => setFormData({ ...formData, experienceMonths: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    required={!isLogin}
                  >
                    <option value="0">Beginner (0-4 months)</option>
                    <option value="5">Intermediate (5-8 months)</option>
                    <option value="9">Advanced (9+ months)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-300 mb-2">Gender</label>
                  <select
                    value={formData.gender}
                    onChange={(e) => setFormData({ ...formData, gender: e.target.value as any })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    required={!isLogin}
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-gray-300 mb-2">Preferred Gym Timing</label>
                  <select
                    value={formData.gymTiming}
                    onChange={(e) => setFormData({ ...formData, gymTiming: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    required={!isLogin}
                  >
                    <option value="morning">Morning</option>
                    <option value="afternoon">Afternoon</option>
                    <option value="evening">Evening</option>
                  </select>
                </div>
              </>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-3 rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  {isLogin ? 'Logging in...' : 'Signing up...'}
                </>
              ) : (
                isLogin ? 'Login' : 'Sign Up'
              )}
            </button>
          </form>

          <p className="text-center text-gray-400 mt-6">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button
              onClick={() => { setIsLogin(!isLogin); setError(null); }}
              className="text-blue-400 hover:text-blue-300"
            >
              {isLogin ? 'Sign Up' : 'Login'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}