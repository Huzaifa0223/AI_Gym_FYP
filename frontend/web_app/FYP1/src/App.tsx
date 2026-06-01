import { useState, useEffect } from 'react';
import { LandingPage } from './components/LandingPage';
import { LoginSignup } from './components/LoginSignup';
import { Dashboard } from './components/Dashboard';
import { WorkoutSelection } from './components/WorkoutSelection';
import { CameraWorkout } from './components/CameraWorkout';
import { ManualWorkout } from './components/ManualWorkout';
import { WorkoutSummary } from './components/WorkoutSummary';
import { Profile } from './components/Profile';
import { authAPI, workoutAPI, setToken, getToken, type User } from './services/api';
import { exerciseDatabase } from './data/exercises';

type Screen = 'landing' | 'login' | 'dashboard' | 'workout-selection' | 'camera' | 'manual' | 'summary' | 'profile';

interface WorkoutSession {
  exerciseName: string;
  reps: number;
  accuracy: number;
  mistakes: string[];
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('landing');
  const [user, setUser] = useState<User | null>(null);
  const [selectedExercise, setSelectedExercise] = useState<string>('');
  const [selectedMuscleGroup, setSelectedMuscleGroup] = useState<string>('');
  const [workoutSession, setWorkoutSession] = useState<WorkoutSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing token on app load
  useEffect(() => {
    const verifyAuth = async () => {
      const token = getToken();
      if (token) {
        try {
          const response = await authAPI.verify();
          if (response.valid && response.user) {
            setUser(response.user);
            setCurrentScreen('dashboard');
          }
        } catch (error) {
          // Token invalid, clear it
          setToken(null);
        }
      }
      setIsLoading(false);
    };

    verifyAuth();
  }, []);

  const handleGetStarted = () => {
    setCurrentScreen('login');
  };

  const handleLogin = (userData: User) => {
    setUser(userData);
    setCurrentScreen('dashboard');
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    setCurrentScreen('login');
  };

  const handleStartWorkout = () => {
    setCurrentScreen('workout-selection');
  };

  const handleExerciseSelect = (
    exercise: string,
    mode: 'camera' | 'manual',
    muscleGroup = '',
  ) => {
    setSelectedExercise(exercise);
    setSelectedMuscleGroup(muscleGroup);  // authoritative for ML routing in CameraWorkout
    setCurrentScreen(mode);
  };

  const handleWorkoutComplete = async (session: WorkoutSession) => {
    setWorkoutSession(session);
    setCurrentScreen('summary');

    const exerciseData = exerciseDatabase.find(
      e => e.name.toLowerCase() === session.exerciseName.toLowerCase()
    );
    const muscleGroup = exerciseData?.muscleGroup || 'Unknown';

    try {
      await workoutAPI.addWorkout({
        exerciseName: session.exerciseName,
        muscleGroup,
        reps: session.reps,
        sets: 1,
        accuracy: session.accuracy,
        workoutDate: new Date().toISOString().split('T')[0],
      });
    } catch (error) {
      console.error('Failed to save workout to database:', error);
    }
  };

  const handleBackToDashboard = () => {
    setCurrentScreen('dashboard');
  };

  const handleViewProfile = () => {
    setCurrentScreen('profile');
  };

  const handleUpdateUser = (updatedUser: User) => {
    setUser(updatedUser);
  };

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {currentScreen === 'landing' && (
        <LandingPage onGetStarted={handleGetStarted} />
      )}

      {currentScreen === 'login' && (
        <LoginSignup onLogin={handleLogin} />
      )}

      {currentScreen === 'dashboard' && user && (
        <Dashboard
          user={user}
          onStartWorkout={handleStartWorkout}
          onViewProfile={handleViewProfile}
          onLogout={handleLogout}
        />
      )}

      {currentScreen === 'workout-selection' && user && (
        <WorkoutSelection
          user={user}
          onExerciseSelect={handleExerciseSelect}
          onBack={handleBackToDashboard}
        />
      )}

      {currentScreen === 'camera' && selectedExercise && (
        <CameraWorkout
          exercise={selectedExercise}
          muscleGroup={selectedMuscleGroup}
          userAge={user?.age}
          onComplete={handleWorkoutComplete}
          onBack={() => setCurrentScreen('workout-selection')}
        />
      )}

      {currentScreen === 'manual' && selectedExercise && (
        <ManualWorkout
          exercise={selectedExercise}
          onComplete={handleWorkoutComplete}
          onBack={() => setCurrentScreen('workout-selection')}
        />
      )}

      {currentScreen === 'summary' && workoutSession && (
        <WorkoutSummary
          session={workoutSession}
          onFinish={() => setCurrentScreen('dashboard')}
        />
      )}

      {currentScreen === 'profile' && user && (
        <Profile
          user={user}
          onBack={handleBackToDashboard}
          onUpdateUser={handleUpdateUser}
        />
      )}
    </div>
  );
}