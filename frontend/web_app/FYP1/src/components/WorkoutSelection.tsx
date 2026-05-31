import { useState } from 'react';
import { ArrowLeft, Search, Target, TrendingUp } from 'lucide-react';
import type { User } from '../services/api';
import { getFilteredExercises, weeklySchedule, getAgeGroup, getLevel, type Exercise, type MuscleGroup } from '../data/exercises';

interface WorkoutSelectionProps {
  user: User;
  onExerciseSelect: (exerciseName: string, mode: 'camera' | 'manual') => void;
  onBack: () => void;
}

export function WorkoutSelection({ user, onExerciseSelect, onBack }: WorkoutSelectionProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Use user's custom schedule if available
  const userSchedule = user.weeklySchedule || weeklySchedule;
  const todayDay = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][new Date().getDay()];
  const todaysMuscle = userSchedule[todayDay] || 'Rest';

  const userAgeGroup = getAgeGroup(user.age);
  const userLevel = getLevel(user.experienceMonths);

  // Handle multiple muscle groups (e.g., "Biceps/Triceps")
  const muscleGroups = todaysMuscle.split('/');

  // Get exercises for today's muscle group(s)
  let exercises: Exercise[] = [];
  if (todaysMuscle !== 'Rest') {
    muscleGroups.forEach(muscleGroup => {
      const muscleExercises = getFilteredExercises(muscleGroup.trim() as MuscleGroup, user.goal, userLevel, userAgeGroup);
      exercises = [...exercises, ...muscleExercises];
    });
  }

  // Filter exercises by search query
  const filteredExercises = exercises.filter(exercise =>
    exercise.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    exercise.equipment.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getLevelText = (level: string) => {
    switch (level) {
      case 'easy': return 'Beginner';
      case 'intermediate': return 'Intermediate';
      case 'hard': return 'Advanced';
      default: return level;
    }
  };

  const getGoalText = (goal: string) => {
    switch (goal) {
      case 'gain': return 'Muscle Gain';
      case 'loss': return 'Weight Loss';
      case 'maintenance': return 'Maintenance';
      default: return goal;
    }
  };

  if (todaysMuscle === 'Rest') {
    return (
      <div className="min-h-screen p-4 md:p-8">
        <div className="max-w-7xl mx-auto">
          <button
            onClick={onBack}
            className="mb-6 p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all flex items-center gap-2"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Dashboard
          </button>

          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
              <Target className="w-12 h-12 text-gray-400" />
            </div>
            <h2 className="text-white text-3xl mb-4">It's Rest Day!</h2>
            <p className="text-gray-400 text-lg max-w-md mx-auto">
              Recovery is an essential part of your fitness journey. Take today to rest and let your muscles recover for tomorrow's workout.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <button
            onClick={onBack}
            className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-white text-3xl">Choose Your Exercise</h1>
            <p className="text-gray-400">Today's focus: {todaysMuscle}</p>
          </div>
        </div>

        {/* User Info Banner */}
        <div className="mb-6 bg-blue-500/20 backdrop-blur-lg rounded-xl p-4 border border-blue-500/30">
          <div className="flex flex-wrap items-center gap-6">
            <div>
              <p className="text-gray-300">Level</p>
              <p className="text-white">{getLevelText(userLevel)}</p>
            </div>
            <div className="w-px h-8 bg-white/20" />
            <div>
              <p className="text-gray-300">Goal</p>
              <p className="text-white">{getGoalText(user.goal)}</p>
            </div>
            <div className="w-px h-8 bg-white/20" />
            <div>
              <p className="text-gray-300">Age Group</p>
              <p className="text-white">
                {user.age <= 16 ? '10-16 years' : user.age <= 45 ? '17-45 years' : '46+ years'}
              </p>
            </div>
            <div className="w-px h-8 bg-white/20" />
            <div>
              <p className="text-gray-300">Exercises Today</p>
              <p className="text-white">{userLevel === 'hard' ? '8 exercises' : '6 exercises'}</p>
            </div>
          </div>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search exercises..."
              className="w-full bg-white/10 backdrop-blur-lg border border-white/20 rounded-xl px-12 py-4 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>

        {/* Exercise Count */}
        <div className="mb-4">
          <p className="text-gray-400">
            Showing {filteredExercises.length} {userLevel === 'easy' ? 'beginner-friendly' : userLevel === 'intermediate' ? 'intermediate' : 'advanced'} exercises for {todaysMuscle}
          </p>
        </div>

        {/* Exercise Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredExercises.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <p className="text-gray-400">No exercises found matching your search.</p>
            </div>
          ) : (
            filteredExercises.map((exercise) => (
              <ExerciseCard
                key={exercise.name}
                exercise={exercise}
                onSelect={(mode) => onExerciseSelect(exercise.name, mode)}
              />
            ))
          )}
        </div>

        {filteredExercises.length > 0 && (
          <div className="mt-8 bg-cyan-500/10 backdrop-blur-lg rounded-xl p-4 border border-cyan-500/30">
            <p className="text-cyan-400">
              💡 <strong>Tip:</strong> {userLevel === 'easy'
                ? 'Focus on proper form rather than heavy weights. Take your time with each rep.'
                : userLevel === 'intermediate'
                  ? 'Increase weight gradually and maintain good form throughout each set.'
                  : 'Push your limits while maintaining perfect form. Consider adding drop sets for extra intensity.'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

interface ExerciseCardProps {
  exercise: Exercise;
  onSelect: (mode: 'camera' | 'manual') => void;
}

function ExerciseCard({ exercise, onSelect }: ExerciseCardProps) {
  const [showModeSelection, setShowModeSelection] = useState(false);

  const difficultyColors = {
    easy: 'bg-green-500/20 text-green-400 border border-green-500/30',
    intermediate: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
    hard: 'bg-red-500/20 text-red-400 border border-red-500/30'
  };

  return (
    <>
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:border-blue-500/50 transition-all group">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-white">{exercise.name}</h3>
            </div>
            <span className={`text-xs px-2 py-1 rounded ${difficultyColors[exercise.level]}`}>
              {exercise.level.charAt(0).toUpperCase() + exercise.level.slice(1)}
            </span>
          </div>
          <Target className="w-5 h-5 text-blue-400" />
        </div>

        <p className="text-gray-400 mb-4">{exercise.description}</p>

        <div className="space-y-2 mb-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Equipment:</span>
            <span className="text-gray-300">{exercise.equipment}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Reps:</span>
            <span className="text-gray-300">{exercise.reps}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Sets:</span>
            <span className="text-gray-300">{exercise.sets}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Rest:</span>
            <span className="text-gray-300">{exercise.restTime}s</span>
          </div>
        </div>

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-gray-400" />
            <span className="text-gray-400">~{exercise.calories} cal</span>
          </div>
        </div>

        {/* Form Tips */}
        {exercise.formTips.length > 0 && (
          <div className="mb-4 bg-black/30 rounded-lg p-3">
            <p className="text-blue-400 mb-2">Form Tips:</p>
            <ul className="space-y-1">
              {exercise.formTips.slice(0, 2).map((tip, index) => (
                <li key={index} className="text-gray-300 flex items-start gap-2">
                  <span className="w-1 h-1 bg-blue-400 rounded-full mt-2 flex-shrink-0" />
                  <span className="text-xs">{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <button
          onClick={() => setShowModeSelection(true)}
          className="w-full bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 transition-all group-hover:shadow-lg group-hover:shadow-blue-500/50"
        >
          Start Exercise
        </button>
      </div>

      {/* Mode Selection Modal */}
      {showModeSelection && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 rounded-2xl p-6 max-w-md w-full border border-white/20">
            <h3 className="text-white text-2xl mb-4">Choose Workout Mode</h3>
            <p className="text-gray-400 mb-6">How would you like to track this exercise?</p>

            <div className="space-y-3">
              <button
                onClick={() => {
                  setShowModeSelection(false);
                  onSelect('camera');
                }}
                className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-4 px-6 rounded-xl hover:from-blue-600 hover:to-cyan-600 transition-all text-left"
              >
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Target className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="font-medium">Camera Mode</p>
                    <p className="text-sm text-blue-100">AI tracks your form in real-time</p>
                  </div>
                </div>
              </button>

              <button
                onClick={() => {
                  setShowModeSelection(false);
                  onSelect('manual');
                }}
                className="w-full bg-white/10 text-white py-4 px-6 rounded-xl hover:bg-white/20 transition-all text-left border border-white/20"
              >
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-white/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <TrendingUp className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="font-medium">Manual Mode</p>
                    <p className="text-sm text-gray-400">Track reps and sets yourself</p>
                  </div>
                </div>
              </button>

              <button
                onClick={() => setShowModeSelection(false)}
                className="w-full text-gray-400 py-2 hover:text-white transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}