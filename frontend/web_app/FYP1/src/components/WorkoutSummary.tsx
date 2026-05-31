import { useEffect, useState } from 'react';
import { Trophy, TrendingUp, AlertTriangle, Home, Share2, CheckCircle } from 'lucide-react';
import { workoutAPI, achievementAPI } from '../services/api';

interface WorkoutSummaryProps {
  session: {
    exerciseName: string;
    reps: number;
    accuracy: number;
    mistakes: string[];
    muscleGroup?: string;
    sets?: number;
    durationMinutes?: number;
  };
  onFinish: () => void;
}

export function WorkoutSummary({ session, onFinish }: WorkoutSummaryProps) {
  const [isSaving, setIsSaving] = useState(true);
  const [saved, setSaved] = useState(false);
  const [newAchievements, setNewAchievements] = useState<string[]>([]);

  // Save workout to API when component mounts
  useEffect(() => {
    const saveWorkout = async () => {
      try {
        const estimatedCalories = Math.round(session.reps * 2.5);

        await workoutAPI.addWorkout({
          exerciseName: session.exerciseName,
          muscleGroup: session.muscleGroup || 'Unknown',
          reps: session.reps,
          sets: session.sets || 1,
          accuracy: session.accuracy,
          durationMinutes: session.durationMinutes || 5,
          caloriesBurned: estimatedCalories
        });

        // Check for new achievements
        const achievementRes = await achievementAPI.checkAchievements();
        if (achievementRes.newAchievements.length > 0) {
          setNewAchievements(achievementRes.newAchievements.map(a => a.title));
        }

        setSaved(true);
      } catch (error) {
        console.error('Failed to save workout:', error);
      }
      setIsSaving(false);
    };

    saveWorkout();
  }, [session]);

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 90) return 'text-green-400';
    if (accuracy >= 75) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getAccuracyBgColor = (accuracy: number) => {
    if (accuracy >= 90) return 'from-green-500/20 to-green-600/20 border-green-500/30';
    if (accuracy >= 75) return 'from-yellow-500/20 to-yellow-600/20 border-yellow-500/30';
    return 'from-red-500/20 to-red-600/20 border-red-500/30';
  };

  const estimatedCalories = Math.round(session.reps * 2.5);

  return (
    <div className="min-h-screen p-4 md:p-8 flex items-center justify-center">
      <div className="max-w-2xl w-full">
        {/* Success Icon */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full mb-4 animate-bounce">
            <Trophy className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-white text-4xl mb-2">Workout Complete!</h1>
          <p className="text-gray-400">Great job on finishing your session</p>

          {/* Saving status */}
          <div className="mt-3 flex items-center justify-center gap-2">
            {isSaving ? (
              <>
                <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-blue-400 text-sm">Saving your workout...</span>
              </>
            ) : saved ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-green-400 text-sm">Workout saved successfully!</span>
              </>
            ) : null}
          </div>
        </div>

        {/* New Achievements */}
        {newAchievements.length > 0 && (
          <div className="mb-6 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 backdrop-blur-lg rounded-xl p-6 border border-yellow-500/30">
            <div className="flex items-center gap-2 mb-3">
              <Trophy className="w-6 h-6 text-yellow-400" />
              <h3 className="text-yellow-400 text-xl">New Achievement{newAchievements.length > 1 ? 's' : ''} Unlocked!</h3>
            </div>
            <div className="space-y-2">
              {newAchievements.map((achievement, index) => (
                <div key={index} className="flex items-center gap-2 text-white">
                  <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
                  <span>{achievement}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Stats Card */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 mb-6">
          <h2 className="text-white text-2xl mb-6">{session.exerciseName}</h2>

          {/* Main Stats Grid */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-black/30 rounded-xl p-4 text-center">
              <p className="text-gray-400 mb-2">Reps Completed</p>
              <p className="text-white text-3xl">{session.reps}</p>
            </div>

            <div className={`bg-gradient-to-br ${getAccuracyBgColor(session.accuracy)} rounded-xl p-4 text-center border`}>
              <p className="text-gray-300 mb-2">Accuracy</p>
              <p className={`text-3xl ${getAccuracyColor(session.accuracy)}`}>
                {session.accuracy}%
              </p>
            </div>

            <div className="bg-black/30 rounded-xl p-4 text-center">
              <p className="text-gray-400 mb-2">Est. Calories</p>
              <p className="text-white text-3xl">{estimatedCalories}</p>
            </div>
          </div>

          {/* Accuracy Progress Bar */}
          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <span className="text-gray-400">Form Accuracy</span>
              <span className={getAccuracyColor(session.accuracy)}>{session.accuracy}%</span>
            </div>
            <div className="w-full h-3 bg-black/30 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${session.accuracy >= 90
                    ? 'bg-gradient-to-r from-green-400 to-green-500'
                    : session.accuracy >= 75
                      ? 'bg-gradient-to-r from-yellow-400 to-yellow-500'
                      : 'bg-gradient-to-r from-red-400 to-red-500'
                  }`}
                style={{ width: `${session.accuracy}%` }}
              />
            </div>
          </div>

          {/* Mistakes Section */}
          {session.mistakes.length > 0 && (
            <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-5 h-5 text-orange-400" />
                <h3 className="text-orange-400">Areas for Improvement</h3>
              </div>
              <ul className="space-y-2">
                {[...new Set(session.mistakes)].map((mistake, index) => (
                  <li key={index} className="text-gray-300 flex items-start gap-2">
                    <span className="w-1.5 h-1.5 bg-orange-400 rounded-full mt-2 flex-shrink-0" />
                    <span>{mistake}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {session.mistakes.length === 0 && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 text-center">
              <p className="text-green-400">Perfect form! No mistakes detected! 🎯</p>
            </div>
          )}
        </div>

        {/* Insights */}
        <div className="bg-blue-500/10 backdrop-blur-lg rounded-xl p-6 border border-blue-500/30 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            <h3 className="text-blue-400">AI Insights</h3>
          </div>
          <ul className="space-y-2 text-gray-300">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
              <span>
                {session.accuracy >= 85
                  ? 'Excellent form consistency throughout the workout'
                  : 'Focus on maintaining form throughout the entire exercise'}
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
              <span>
                {session.reps >= 15
                  ? 'Great endurance! Consider increasing difficulty'
                  : 'Good progress! Aim for more reps next time'}
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
              <span>Next workout: Try to maintain {session.accuracy >= 85 ? 'this level' : 'better'} form accuracy</span>
            </li>
          </ul>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <button
            onClick={onFinish}
            disabled={isSaving}
            className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-4 rounded-xl hover:from-blue-600 hover:to-cyan-600 transition-all shadow-lg hover:shadow-blue-500/50 flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Home className="w-5 h-5" />
            Back to Dashboard
          </button>

          <button
            onClick={() => {
              // Mock share functionality
              alert('Share functionality would be implemented here!');
            }}
            className="bg-white/10 backdrop-blur-lg border border-white/20 text-white px-6 py-4 rounded-xl hover:bg-white/20 transition-all"
          >
            <Share2 className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
}