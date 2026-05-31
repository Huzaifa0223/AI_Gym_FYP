import { useState } from 'react';
import { ArrowLeft, CheckCircle, PlusCircle, MinusCircle, Clock } from 'lucide-react';

interface ManualWorkoutProps {
  exercise: string;
  onComplete: (session: {
    exerciseName: string;
    reps: number;
    sets: number;
    accuracy: number;
    mistakes: string[];
  }) => void;
  onBack: () => void;
}

export function ManualWorkout({ exercise, onComplete, onBack }: ManualWorkoutProps) {
  const [sets, setSets] = useState<{ reps: number; completed: boolean }[]>([
    { reps: 0, completed: false },
    { reps: 0, completed: false },
    { reps: 0, completed: false }
  ]);
  const [currentSetIndex, setCurrentSetIndex] = useState(0);
  const [restTime, setRestTime] = useState(60);
  const [isResting, setIsResting] = useState(false);

  const handleAddSet = () => {
    setSets([...sets, { reps: 0, completed: false }]);
  };

  const handleRemoveSet = (index: number) => {
    if (sets.length > 1) {
      const newSets = sets.filter((_, i) => i !== index);
      setSets(newSets);
      if (currentSetIndex >= newSets.length) {
        setCurrentSetIndex(newSets.length - 1);
      }
    }
  };

  const handleRepsChange = (index: number, reps: number) => {
    const newSets = [...sets];
    newSets[index].reps = Math.max(0, reps);
    setSets(newSets);
  };

  const handleCompleteSet = (index: number) => {
    const newSets = [...sets];
    newSets[index].completed = true;
    setSets(newSets);
    
    // Move to next set
    if (index < sets.length - 1) {
      setCurrentSetIndex(index + 1);
      setIsResting(true);
      
      // Auto-advance after rest time
      setTimeout(() => {
        setIsResting(false);
      }, restTime * 1000);
    }
  };

  const handleFinishWorkout = () => {
    const totalReps = sets.reduce((sum, set) => sum + set.reps, 0);
    const completedSets = sets.filter(set => set.completed).length;
    
    // Calculate a simple "accuracy" based on completion
    const accuracy = Math.round((completedSets / sets.length) * 100);
    
    onComplete({
      exerciseName: exercise,
      reps: totalReps,
      sets: completedSets,
      accuracy: accuracy,
      mistakes: [] // Manual mode doesn't track mistakes
    });
  };

  const allSetsCompleted = sets.every(set => set.completed);
  const totalReps = sets.reduce((sum, set) => sum + set.reps, 0);
  const completedSets = sets.filter(set => set.completed).length;

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBack}
            className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="text-center flex-1 mx-4">
            <h1 className="text-white text-3xl mb-1">{exercise}</h1>
            <p className="text-gray-400">Manual Tracking Mode</p>
          </div>
          <div className="w-12" />
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-500/20 backdrop-blur-lg rounded-xl p-4 border border-blue-500/30 text-center">
            <p className="text-blue-400 mb-1">Total Sets</p>
            <p className="text-white text-2xl">{sets.length}</p>
          </div>
          <div className="bg-cyan-500/20 backdrop-blur-lg rounded-xl p-4 border border-cyan-500/30 text-center">
            <p className="text-cyan-400 mb-1">Completed</p>
            <p className="text-white text-2xl">{completedSets}/{sets.length}</p>
          </div>
          <div className="bg-green-500/20 backdrop-blur-lg rounded-xl p-4 border border-green-500/30 text-center">
            <p className="text-green-400 mb-1">Total Reps</p>
            <p className="text-white text-2xl">{totalReps}</p>
          </div>
        </div>

        {/* Rest Timer */}
        {isResting && (
          <div className="mb-6 bg-orange-500/20 backdrop-blur-lg rounded-xl p-6 border border-orange-500/30 text-center animate-pulse">
            <Clock className="w-8 h-8 text-orange-400 mx-auto mb-2" />
            <p className="text-orange-400">Rest Time</p>
            <p className="text-white text-xl">{restTime} seconds</p>
          </div>
        )}

        {/* Sets List */}
        <div className="space-y-4 mb-6">
          {sets.map((set, index) => (
            <div
              key={index}
              className={`bg-white/10 backdrop-blur-lg rounded-xl p-6 border transition-all ${
                set.completed
                  ? 'border-green-500/50 bg-green-500/10'
                  : currentSetIndex === index
                  ? 'border-blue-500/50 bg-blue-500/10'
                  : 'border-white/20'
              }`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    set.completed ? 'bg-green-500' : 'bg-gray-500/30'
                  }`}>
                    {set.completed ? (
                      <CheckCircle className="w-6 h-6 text-white" />
                    ) : (
                      <span className="text-white">{index + 1}</span>
                    )}
                  </div>
                  <div>
                    <h3 className="text-white">Set {index + 1}</h3>
                    <p className="text-gray-400">
                      {set.completed ? `Completed: ${set.reps} reps` : 'In Progress'}
                    </p>
                  </div>
                </div>
                
                {!set.completed && sets.length > 1 && (
                  <button
                    onClick={() => handleRemoveSet(index)}
                    className="p-2 bg-red-500/20 rounded-lg text-red-400 hover:bg-red-500/30 transition-all"
                  >
                    <MinusCircle className="w-5 h-5" />
                  </button>
                )}
              </div>

              {!set.completed && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-gray-300 mb-2">Reps Completed</label>
                    <div className="flex items-center gap-4">
                      <button
                        onClick={() => handleRepsChange(index, set.reps - 1)}
                        className="p-3 bg-white/10 rounded-lg text-white hover:bg-white/20 transition-all"
                      >
                        <MinusCircle className="w-5 h-5" />
                      </button>
                      <input
                        type="number"
                        value={set.reps}
                        onChange={(e) => handleRepsChange(index, parseInt(e.target.value) || 0)}
                        className="flex-1 bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white text-center focus:outline-none focus:border-blue-500"
                        min="0"
                      />
                      <button
                        onClick={() => handleRepsChange(index, set.reps + 1)}
                        className="p-3 bg-white/10 rounded-lg text-white hover:bg-white/20 transition-all"
                      >
                        <PlusCircle className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  <button
                    onClick={() => handleCompleteSet(index)}
                    disabled={set.reps === 0}
                    className={`w-full py-3 rounded-lg transition-all ${
                      set.reps === 0
                        ? 'bg-gray-500/30 text-gray-500 cursor-not-allowed'
                        : 'bg-green-500 text-white hover:bg-green-600'
                    }`}
                  >
                    Complete Set
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Add Set Button */}
        {!allSetsCompleted && (
          <button
            onClick={handleAddSet}
            className="w-full py-4 bg-white/10 border-2 border-dashed border-white/30 rounded-xl text-white hover:bg-white/20 hover:border-blue-500/50 transition-all mb-6 flex items-center justify-center gap-2"
          >
            <PlusCircle className="w-5 h-5" />
            Add Another Set
          </button>
        )}

        {/* Rest Time Setting */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 mb-6">
          <label className="block text-gray-300 mb-3">Rest Time Between Sets (seconds)</label>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setRestTime(Math.max(15, restTime - 15))}
              className="p-2 bg-white/10 rounded-lg text-white hover:bg-white/20 transition-all"
            >
              <MinusCircle className="w-5 h-5" />
            </button>
            <input
              type="number"
              value={restTime}
              onChange={(e) => setRestTime(Math.max(15, parseInt(e.target.value) || 60))}
              className="flex-1 bg-black/30 border border-white/20 rounded-lg px-4 py-2 text-white text-center focus:outline-none focus:border-blue-500"
              min="15"
              step="15"
            />
            <button
              onClick={() => setRestTime(restTime + 15)}
              className="p-2 bg-white/10 rounded-lg text-white hover:bg-white/20 transition-all"
            >
              <PlusCircle className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Finish Workout Button */}
        <button
          onClick={handleFinishWorkout}
          disabled={completedSets === 0}
          className={`w-full py-4 rounded-xl transition-all shadow-lg ${
            completedSets === 0
              ? 'bg-gray-500/30 text-gray-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600 hover:shadow-blue-500/50'
          }`}
        >
          {allSetsCompleted ? 'Finish Workout' : `Finish Early (${completedSets} sets completed)`}
        </button>
      </div>
    </div>
  );
}
