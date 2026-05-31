import { useState, useEffect } from 'react';
import { Dumbbell, Calendar, Target, Flame, Activity, TrendingUp, LogOut, User as UserIcon } from 'lucide-react';
import { userAPI, workoutAPI, type User, type WorkoutStats, type Workout } from '../services/api';
import { weeklySchedule as defaultSchedule } from '../data/exercises';

interface DashboardProps {
  user: User;
  onStartWorkout: () => void;
  onViewProfile: () => void;
  onLogout: () => void;
}

export function Dashboard({ user, onStartWorkout, onViewProfile, onLogout }: DashboardProps) {
  const [stats, setStats] = useState<WorkoutStats>({
    totalWorkouts: 0,
    weeklyWorkouts: 0,
    averageAccuracy: 0,
    caloriesBurned: 0,
    totalReps: 0,
    totalDuration: 0
  });
  const [recentWorkouts, setRecentWorkouts] = useState<Workout[]>([]);
  const [userSchedule, setUserSchedule] = useState<Record<string, string>>(user.weeklySchedule || defaultSchedule);

  // Load stats and recent workouts from API
  useEffect(() => {
    const loadData = async () => {
      try {
        const [statsRes, workoutsRes, scheduleRes] = await Promise.all([
          workoutAPI.getStats(),
          workoutAPI.getWorkouts({ limit: 3 }),
          userAPI.getSchedule()
        ]);
        setStats(statsRes);
        setRecentWorkouts(workoutsRes.workouts);
        setUserSchedule(scheduleRes);
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      }
    };
    loadData();
  }, []);

  const todayDay = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][new Date().getDay()];
  const todaysMuscle = userSchedule[todayDay] || 'Rest';

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';

    const diffDays = Math.floor((today.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    return `${diffDays} days ago`;
  };

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-white text-3xl mb-1">Welcome back, {user.name}!</h1>
            <p className="text-gray-400">Ready to crush your workout today?</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onViewProfile}
              className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
            >
              <UserIcon className="w-5 h-5" />
            </button>
            <button
              onClick={onLogout}
              className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Today's Focus */}
        <div className="mb-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 mb-1">{todayDay}'s Focus</p>
              <h2 className="text-white text-3xl">{todaysMuscle}</h2>
              {todaysMuscle === 'Rest' ? (
                <p className="text-blue-100 mt-2">Recovery is just as important as training!</p>
              ) : (
                <p className="text-blue-100 mt-2">Let's build those {todaysMuscle.toLowerCase()} muscles!</p>
              )}
            </div>
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
              <Dumbbell className="w-8 h-8 text-white" />
            </div>
          </div>
        </div>

        {/* Weekly Schedule */}
        <div className="mb-8">
          <h3 className="text-white text-xl mb-4">Weekly Training Schedule</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
            {Object.entries(userSchedule).map(([day, muscle]) => {
              const isToday = day === todayDay;
              const isRest = muscle === 'Rest';
              return (
                <div
                  key={day}
                  className={`p-4 rounded-xl border transition-all ${isToday
                    ? 'bg-blue-500/30 border-blue-500/50'
                    : isRest
                      ? 'bg-gray-500/10 border-gray-500/30'
                      : 'bg-white/10 border-white/20'
                    }`}
                >
                  <p className={`text-xs mb-2 ${isToday ? 'text-blue-300' : 'text-gray-400'}`}>
                    {day.substring(0, 3)}
                  </p>
                  <p className={`${isToday ? 'text-white' : isRest ? 'text-gray-300' : 'text-gray-200'}`}>
                    {muscle}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-lg rounded-2xl p-6 border border-blue-500/30">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/30 rounded-xl">
                <Dumbbell className="w-6 h-6 text-blue-300" />
              </div>
            </div>
            <p className="text-gray-400 mb-1">Total Workouts</p>
            <p className="text-white text-3xl">{stats.totalWorkouts}</p>
          </div>

          <div className="bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 backdrop-blur-lg rounded-2xl p-6 border border-cyan-500/30">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-cyan-500/30 rounded-xl">
                <Calendar className="w-6 h-6 text-cyan-300" />
              </div>
            </div>
            <p className="text-gray-400 mb-1">This Week</p>
            <p className="text-white text-3xl">{stats.weeklyWorkouts}</p>
          </div>

          <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-lg rounded-2xl p-6 border border-green-500/30">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/30 rounded-xl">
                <Target className="w-6 h-6 text-green-300" />
              </div>
            </div>
            <p className="text-gray-400 mb-1">Avg Accuracy</p>
            <p className="text-white text-3xl">{stats.averageAccuracy}%</p>
          </div>

          <div className="bg-gradient-to-br from-orange-500/20 to-orange-600/20 backdrop-blur-lg rounded-2xl p-6 border border-orange-500/30">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-500/30 rounded-xl">
                <Flame className="w-6 h-6 text-orange-300" />
              </div>
            </div>
            <p className="text-gray-400 mb-1">Calories Burned</p>
            <p className="text-white text-3xl">{stats.caloriesBurned}</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Start Workout Card */}
          <div className="lg:col-span-2">
            <div className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl p-8 shadow-2xl">
              <div className="flex items-center gap-4 mb-6">
                <div className="p-4 bg-white/20 rounded-2xl">
                  <Activity className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-white text-2xl">Ready to Workout?</h2>
                  <p className="text-blue-100">Let AI guide your form in real-time</p>
                </div>
              </div>

              <button
                onClick={onStartWorkout}
                disabled={todaysMuscle === 'Rest'}
                className={`w-full py-4 rounded-xl transition-all shadow-lg ${todaysMuscle === 'Rest'
                  ? 'bg-gray-500 text-gray-300 cursor-not-allowed'
                  : 'bg-white text-blue-600 hover:bg-gray-100'
                  }`}
              >
                {todaysMuscle === 'Rest' ? 'Rest Day - No Workout Today' : 'Start New Workout'}
              </button>

              <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                  <p className="text-blue-100 mb-1">Your Goal</p>
                  <p className="text-white capitalize">{user.goal === 'gain' ? 'Gain Weight' : user.goal === 'loss' ? 'Lose Weight' : 'Maintenance'}</p>
                </div>
                <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                  <p className="text-blue-100 mb-1">Age Group</p>
                  <p className="text-white">
                    {user.age <= 16 ? '10-16' : user.age <= 45 ? '17-45' : '46+'}
                  </p>
                </div>
                <div className="bg-white/10 backdrop-blur rounded-xl p-4">
                  <p className="text-blue-100 mb-1">Level</p>
                  <p className="text-white">
                    {user.experienceMonths <= 4 ? 'Beginner' : user.experienceMonths <= 8 ? 'Intermediate' : 'Advanced'}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <div className="flex items-center gap-3 mb-6">
              <TrendingUp className="w-6 h-6 text-blue-400" />
              <h3 className="text-white">Recent Activity</h3>
            </div>

            <div className="space-y-4">
              {recentWorkouts.length > 0 ? (
                recentWorkouts.map((workout) => (
                  <div key={workout.id} className="bg-black/30 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-white">{workout.exerciseName}</p>
                      <span className="text-green-400">{workout.accuracy || 0}%</span>
                    </div>
                    <p className="text-gray-400">{formatDate(workout.workoutDate)}</p>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <Dumbbell className="w-10 h-10 text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400">No recent workouts</p>
                  <p className="text-gray-500 text-sm">Start a workout to see activity here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}