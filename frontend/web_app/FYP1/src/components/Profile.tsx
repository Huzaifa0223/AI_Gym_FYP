import { useState, useEffect } from 'react';
import { ArrowLeft, User, Calendar, Weight, Ruler, Target, TrendingUp, Dumbbell, Award, Edit2, Save, X, Settings, Lock } from 'lucide-react';
import { userAPI, workoutAPI, type User as UserType, type Workout, type ChangePasswordData } from '../services/api';

interface ProfileProps {
  user: UserType;
  onBack: () => void;
  onUpdateUser?: (user: UserType) => void;
}

export function Profile({ user, onBack, onUpdateUser }: ProfileProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [isEditingSchedule, setIsEditingSchedule] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);
  const [editedUser, setEditedUser] = useState({ ...user });
  const [customSchedule, setCustomSchedule] = useState(user.weeklySchedule || {
    Monday: 'Chest',
    Tuesday: 'Biceps',
    Wednesday: 'Back',
    Thursday: 'Triceps',
    Friday: 'Abdominal',
    Saturday: 'Legs',
    Sunday: 'Rest'
  });
  const [workoutHistory, setWorkoutHistory] = useState<Workout[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [passwordData, setPasswordData] = useState({ currentPassword: '', newPassword: '', confirmPassword: '' });
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null);

  // Load workout history and schedule on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [workoutsRes, scheduleRes] = await Promise.all([
          workoutAPI.getWorkouts({ limit: 10 }),
          userAPI.getSchedule()
        ]);
        setWorkoutHistory(workoutsRes.workouts);
        setCustomSchedule(scheduleRes);
      } catch (error) {
        console.error('Failed to load profile data:', error);
      }
    };
    loadData();
  }, []);

  const totalWorkouts = workoutHistory.length;
  const averageAccuracy = workoutHistory.length > 0
    ? Math.round(workoutHistory.filter(w => w.accuracy).reduce((sum, w) => sum + (w.accuracy || 0), 0) / workoutHistory.filter(w => w.accuracy).length) || 0
    : 0;
  const totalDuration = workoutHistory.reduce((sum, w) => sum + (w.durationMinutes || 0), 0);
  const totalReps = workoutHistory.reduce((sum, w) => sum + (w.reps || 0), 0);

  const handleSaveProfile = async () => {
    setIsSaving(true);
    try {
      const response = await userAPI.updateProfile(editedUser);
      if (onUpdateUser && response.user) {
        onUpdateUser({ ...response.user, weeklySchedule: customSchedule });
      }
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save profile:', error);
    }
    setIsSaving(false);
  };

  const handleCancelEdit = () => {
    setEditedUser({ ...user });
    setIsEditing(false);
  };

  const handleSaveSchedule = async () => {
    setIsSaving(true);
    try {
      await userAPI.updateSchedule(customSchedule);
      if (onUpdateUser) {
        onUpdateUser({ ...user, weeklySchedule: customSchedule });
      }
      setIsEditingSchedule(false);
    } catch (error) {
      console.error('Failed to save schedule:', error);
    }
    setIsSaving(false);
  };

  const handleChangePassword = async () => {
    setPasswordError(null);
    setPasswordSuccess(null);

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setPasswordError('New passwords do not match');
      return;
    }

    if (passwordData.newPassword.length < 6) {
      setPasswordError('New password must be at least 6 characters');
      return;
    }

    setIsSaving(true);
    try {
      await userAPI.changePassword({
        currentPassword: passwordData.currentPassword,
        newPassword: passwordData.newPassword
      });
      setPasswordSuccess('Password changed successfully!');
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setTimeout(() => {
        setIsChangingPassword(false);
        setPasswordSuccess(null);
      }, 2000);
    } catch (error) {
      setPasswordError(error instanceof Error ? error.message : 'Failed to change password');
    }
    setIsSaving(false);
  };

  const muscleGroupOptions = ['Chest', 'Biceps', 'Back', 'Triceps', 'Abdominal', 'Legs', 'Rest', 'Shoulders', 'Biceps/Triceps'];

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button
            onClick={onBack}
            className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex-1">
            <h1 className="text-white text-3xl">Profile</h1>
            <p className="text-gray-400">Your fitness journey</p>
          </div>
          <div className="flex gap-2">
            {!isEditing && !isChangingPassword && (
              <>
                <button
                  onClick={() => setIsChangingPassword(true)}
                  className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all flex items-center gap-2"
                >
                  <Lock className="w-5 h-5" />
                  <span className="hidden sm:inline">Change Password</span>
                </button>
                <button
                  onClick={() => setIsEditing(true)}
                  className="p-3 bg-blue-500/20 backdrop-blur-lg rounded-xl border border-blue-500/30 text-blue-400 hover:bg-blue-500/30 transition-all flex items-center gap-2"
                >
                  <Edit2 className="w-5 h-5" />
                  <span className="hidden sm:inline">Edit Profile</span>
                </button>
              </>
            )}
          </div>
        </div>

        {/* Change Password Modal */}
        {isChangingPassword && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <div className="bg-slate-900 rounded-2xl p-6 max-w-md w-full border border-white/20">
              <h3 className="text-white text-2xl mb-4">Change Password</h3>

              {passwordError && (
                <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg">
                  <p className="text-red-400 text-sm">{passwordError}</p>
                </div>
              )}

              {passwordSuccess && (
                <div className="mb-4 p-3 bg-green-500/20 border border-green-500/50 rounded-lg">
                  <p className="text-green-400 text-sm">{passwordSuccess}</p>
                </div>
              )}

              <div className="space-y-4">
                <div>
                  <label className="block text-gray-300 mb-2">Current Password</label>
                  <input
                    type="password"
                    value={passwordData.currentPassword}
                    onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    placeholder="Enter current password"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">New Password</label>
                  <input
                    type="password"
                    value={passwordData.newPassword}
                    onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    placeholder="Enter new password"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">Confirm New Password</label>
                  <input
                    type="password"
                    value={passwordData.confirmPassword}
                    onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                    placeholder="Confirm new password"
                  />
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={handleChangePassword}
                  disabled={isSaving}
                  className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-3 rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all disabled:opacity-50"
                >
                  {isSaving ? 'Saving...' : 'Change Password'}
                </button>
                <button
                  onClick={() => {
                    setIsChangingPassword(false);
                    setPasswordError(null);
                    setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
                  }}
                  className="flex-1 bg-white/10 text-white py-3 rounded-lg hover:bg-white/20 transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Profile Info Card */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex flex-col items-center text-center mb-6">
                <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center mb-4">
                  <User className="w-12 h-12 text-white" />
                </div>
                {isEditing ? (
                  <input
                    type="text"
                    value={editedUser.name}
                    onChange={(e) => setEditedUser({ ...editedUser, name: e.target.value })}
                    className="w-full bg-black/30 border border-white/20 rounded-lg px-4 py-2 text-white text-center mb-2 focus:outline-none focus:border-blue-500"
                  />
                ) : (
                  <h2 className="text-white text-2xl mb-1">{user.name}</h2>
                )}
                <p className="text-gray-400">{user.email}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Calendar className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Age</span>
                  </div>
                  {isEditing ? (
                    <input
                      type="number"
                      value={editedUser.age}
                      onChange={(e) => setEditedUser({ ...editedUser, age: parseInt(e.target.value) })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    />
                  ) : (
                    <p className="text-white">{user.age} years</p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Weight className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Weight</span>
                  </div>
                  {isEditing ? (
                    <input
                      type="number"
                      value={editedUser.weight}
                      onChange={(e) => setEditedUser({ ...editedUser, weight: parseInt(e.target.value) })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    />
                  ) : (
                    <p className="text-white">{user.weight} kg</p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Ruler className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Height</span>
                  </div>
                  {isEditing ? (
                    <input
                      type="number"
                      value={editedUser.height}
                      onChange={(e) => setEditedUser({ ...editedUser, height: parseInt(e.target.value) })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    />
                  ) : (
                    <p className="text-white">{user.height} cm</p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Target className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Goal</span>
                  </div>
                  {isEditing ? (
                    <select
                      value={editedUser.goal}
                      onChange={(e) => setEditedUser({ ...editedUser, goal: e.target.value as 'gain' | 'loss' | 'maintenance' })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="gain">Gain</option>
                      <option value="loss">Loss</option>
                      <option value="maintenance">Maintenance</option>
                    </select>
                  ) : (
                    <p className="text-white capitalize">{user.goal}</p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Dumbbell className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Level</span>
                  </div>
                  {isEditing ? (
                    <select
                      value={editedUser.experienceMonths}
                      onChange={(e) => setEditedUser({ ...editedUser, experienceMonths: parseInt(e.target.value) })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="0">Beginner</option>
                      <option value="5">Intermediate</option>
                      <option value="9">Advanced</option>
                    </select>
                  ) : (
                    <p className="text-white">
                      {user.experienceMonths <= 4 ? 'Beginner' : user.experienceMonths <= 8 ? 'Intermediate' : 'Advanced'}
                    </p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <User className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Gender</span>
                  </div>
                  {isEditing ? (
                    <select
                      value={editedUser.gender}
                      onChange={(e) => setEditedUser({ ...editedUser, gender: e.target.value as 'male' | 'female' | 'other' })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other</option>
                    </select>
                  ) : (
                    <p className="text-white capitalize">{user.gender}</p>
                  )}
                </div>

                <div className="bg-black/30 rounded-xl p-4 col-span-2">
                  <div className="flex items-center gap-3 mb-2">
                    <Calendar className="w-5 h-5 text-blue-400" />
                    <span className="text-gray-400">Gym Timing</span>
                  </div>
                  {isEditing ? (
                    <select
                      value={editedUser.gymTiming}
                      onChange={(e) => setEditedUser({ ...editedUser, gymTiming: e.target.value })}
                      className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="morning">Morning</option>
                      <option value="afternoon">Afternoon</option>
                      <option value="evening">Evening</option>
                    </select>
                  ) : (
                    <p className="text-white capitalize">{user.gymTiming}</p>
                  )}
                </div>
              </div>

              {isEditing && (
                <div className="mt-6 flex gap-3">
                  <button
                    onClick={handleSaveProfile}
                    disabled={isSaving}
                    className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-2 rounded-lg hover:from-blue-600 hover:to-cyan-600 transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <Save className="w-4 h-4" />
                    {isSaving ? 'Saving...' : 'Save'}
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="flex-1 bg-white/10 text-white py-2 rounded-lg hover:bg-white/20 transition-all flex items-center justify-center gap-2"
                  >
                    <X className="w-4 h-4" />
                    Cancel
                  </button>
                </div>
              )}
            </div>

            {/* Stats Overview */}
            <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-lg rounded-2xl p-6 border border-blue-500/30">
              <div className="flex items-center gap-2 mb-4">
                <Award className="w-6 h-6 text-blue-400" />
                <h3 className="text-white">Achievements</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Workouts</span>
                  <span className="text-white">{totalWorkouts}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Avg Accuracy</span>
                  <span className="text-green-400">{averageAccuracy}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Time</span>
                  <span className="text-white">{totalDuration} min</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Reps</span>
                  <span className="text-white">{totalReps}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Weekly Schedule */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  <Settings className="w-6 h-6 text-blue-400" />
                  <h3 className="text-white text-xl">Weekly Schedule</h3>
                </div>
                {!isEditingSchedule ? (
                  <button
                    onClick={() => setIsEditingSchedule(true)}
                    className="px-4 py-2 bg-blue-500/20 rounded-lg text-blue-400 hover:bg-blue-500/30 transition-all flex items-center gap-2"
                  >
                    <Edit2 className="w-4 h-4" />
                    Customize
                  </button>
                ) : (
                  <div className="flex gap-2">
                    <button
                      onClick={handleSaveSchedule}
                      disabled={isSaving}
                      className="px-4 py-2 bg-blue-500 rounded-lg text-white hover:bg-blue-600 transition-all flex items-center gap-2 disabled:opacity-50"
                    >
                      <Save className="w-4 h-4" />
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                      onClick={() => {
                        setCustomSchedule(user.weeklySchedule || {});
                        setIsEditingSchedule(false);
                      }}
                      className="px-4 py-2 bg-white/10 rounded-lg text-white hover:bg-white/20 transition-all"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {Object.entries(customSchedule).map(([day, muscle]) => (
                  <div key={day} className="bg-black/30 rounded-xl p-4">
                    <p className="text-gray-400 mb-2">{day}</p>
                    {isEditingSchedule ? (
                      <select
                        value={muscle}
                        onChange={(e) => setCustomSchedule({ ...customSchedule, [day]: e.target.value })}
                        className="w-full bg-black/30 border border-white/20 rounded px-2 py-1 text-white focus:outline-none focus:border-blue-500"
                      >
                        {muscleGroupOptions.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <p className="text-white">{muscle}</p>
                    )}
                  </div>
                ))}
              </div>

              {isEditingSchedule && (
                <div className="mt-4 bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-3">
                  <p className="text-cyan-400 text-sm">
                    💡 Customize your weekly workout schedule. You can assign any muscle group to any day, and even repeat muscle groups multiple times per week.
                  </p>
                </div>
              )}
            </div>

            {/* Workout History */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex items-center gap-2 mb-6">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                <h3 className="text-white text-xl">Workout History</h3>
              </div>

              <div className="space-y-3">
                {workoutHistory.map((workout) => (
                  <div
                    key={workout.id}
                    className="bg-black/30 rounded-xl p-4 border border-white/10 hover:border-blue-500/50 transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-500/20 rounded-lg">
                          <Dumbbell className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                          <h4 className="text-white">{workout.exerciseName}</h4>
                          <p className="text-gray-400">{workout.workoutDate}</p>
                        </div>
                      </div>

                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-gray-400">Accuracy:</span>
                          <span
                            className={
                              (workout.accuracy || 0) >= 90
                                ? 'text-green-400'
                                : (workout.accuracy || 0) >= 75
                                  ? 'text-yellow-400'
                                  : 'text-red-400'
                            }
                          >
                            {workout.accuracy || 0}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-black/20 rounded-lg p-2">
                        <p className="text-gray-400">Reps</p>
                        <p className="text-white">{workout.reps}</p>
                      </div>
                      <div className="bg-black/20 rounded-lg p-2">
                        <p className="text-gray-400">Duration</p>
                        <p className="text-white">{workout.durationMinutes || 0} min</p>
                      </div>
                      <div className="bg-black/20 rounded-lg p-2">
                        <p className="text-gray-400">Calories</p>
                        <p className="text-white">{workout.caloriesBurned || 0}</p>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="mt-3">
                      <div className="w-full h-1.5 bg-black/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${(workout.accuracy || 0) >= 90
                            ? 'bg-green-500'
                            : (workout.accuracy || 0) >= 75
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                            }`}
                          style={{ width: `${workout.accuracy || 0}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {workoutHistory.length === 0 && (
                <div className="text-center py-12">
                  <Dumbbell className="w-16 h-16 text-gray-400 mx-auto mb-4 opacity-50" />
                  <p className="text-gray-400">No workout history yet</p>
                  <p className="text-gray-500">Start your first workout to see it here</p>
                </div>
              )}
            </div>

            {/* Weekly Progress Chart */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-6 h-6 text-blue-400" />
                <h3 className="text-white">Weekly Progress</h3>
              </div>

              <div className="flex items-end justify-between gap-2 h-48">
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day) => {
                  const height = Math.random() * 100 + 20;
                  return (
                    <div key={day} className="flex-1 flex flex-col items-center gap-2">
                      <div className="w-full bg-gradient-to-t from-blue-500 to-cyan-500 rounded-t-lg transition-all hover:opacity-80" style={{ height: `${height}%` }} />
                      <span className="text-gray-400">{day}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}