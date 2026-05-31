// API Service for AI Gym Trainer Backend
// Use current hostname for API to support network access (mobile testing)
const API_BASE_URL = `http://${window.location.hostname}:5001/api`;

// Token management
let authToken: string | null = localStorage.getItem('token');

export function setToken(token: string | null) {
    authToken = token;
    if (token) {
        localStorage.setItem('token', token);
    } else {
        localStorage.removeItem('token');
    }
}

export function getToken(): string | null {
    return authToken || localStorage.getItem('token');
}

// Generic fetch helper with auth
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const token = getToken();

    const headers: HeadersInit = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    };

    if (token) {
        (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers,
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.message || 'API request failed');
    }

    return data;
}

// ==================== AUTH API ====================

export interface SignupData {
    name: string;
    email: string;
    password: string;
    age: number;
    weight: number;
    height: number;
    goal: 'gain' | 'loss' | 'maintenance';
    experienceMonths: number;
    gender: 'male' | 'female' | 'other';
    gymTiming: string;
}

export interface LoginData {
    email: string;
    password: string;
}

export interface AuthResponse {
    message: string;
    token: string;
    user: User;
}

export interface User {
    id: number;
    name: string;
    email: string;
    age: number;
    weight: number;
    height: number;
    goal: 'gain' | 'loss' | 'maintenance';
    experienceMonths: number;
    gender: 'male' | 'female' | 'other';
    gymTiming: string;
    weeklySchedule?: Record<string, string>;
}

export const authAPI = {
    signup: (data: SignupData) =>
        fetchAPI<AuthResponse>('/auth/signup', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    login: (data: LoginData) =>
        fetchAPI<AuthResponse>('/auth/login', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    verify: () =>
        fetchAPI<{ valid: boolean; user: User }>('/auth/verify'),
};

// ==================== USER API ====================

export interface ChangePasswordData {
    currentPassword: string;
    newPassword: string;
}

export const userAPI = {
    getProfile: () =>
        fetchAPI<User>('/users/profile'),

    updateProfile: (data: Partial<User>) =>
        fetchAPI<{ message: string; user: User }>('/users/profile', {
            method: 'PUT',
            body: JSON.stringify(data),
        }),

    changePassword: (data: ChangePasswordData) =>
        fetchAPI<{ message: string }>('/users/change-password', {
            method: 'PUT',
            body: JSON.stringify(data),
        }),

    getSchedule: () =>
        fetchAPI<Record<string, string>>('/users/schedule'),

    updateSchedule: (schedule: Record<string, string>) =>
        fetchAPI<{ message: string; schedule: Record<string, string> }>('/users/schedule', {
            method: 'PUT',
            body: JSON.stringify(schedule),
        }),
};

// ==================== WORKOUT API ====================

export interface Workout {
    id: number;
    exerciseName: string;
    muscleGroup: string;
    reps: number;
    sets: number;
    accuracy: number | null;
    durationMinutes: number | null;
    caloriesBurned: number | null;
    workoutDate: string;
}

export interface WorkoutStats {
    totalWorkouts: number;
    weeklyWorkouts: number;
    averageAccuracy: number;
    caloriesBurned: number;
    totalReps: number;
    totalDuration: number;
}

export interface AddWorkoutData {
    exerciseName: string;
    muscleGroup: string;
    reps: number;
    sets: number;
    accuracy?: number;
    durationMinutes?: number;
    caloriesBurned?: number;
    workoutDate?: string;
}

export const workoutAPI = {
    getWorkouts: (params?: { limit?: number; offset?: number; startDate?: string; endDate?: string }) => {
        const query = params ? '?' + new URLSearchParams(params as Record<string, string>).toString() : '';
        return fetchAPI<{ workouts: Workout[]; total: number }>(`/workouts${query}`);
    },

    getStats: () =>
        fetchAPI<WorkoutStats>('/workouts/stats'),

    addWorkout: (data: AddWorkoutData) =>
        fetchAPI<{ message: string; workout: Workout }>('/workouts', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    getWorkout: (id: number) =>
        fetchAPI<Workout>(`/workouts/${id}`),

    deleteWorkout: (id: number) =>
        fetchAPI<{ message: string }>(`/workouts/${id}`, {
            method: 'DELETE',
        }),
};

// ==================== EXERCISE API ====================

export interface Exercise {
    name: string;
    muscleGroup: string;
    goal: string[];
    level: string;
    ageGroup: string[];
    equipment: string;
    description: string;
    reps: string;
    sets: number;
    restTime: number;
    calories: number;
    formTips: string[];
    isTimeBased?: boolean;
}

export const exerciseAPI = {
    getExercises: (params?: { muscleGroup?: string; goal?: string; age?: number; experienceMonths?: number }) => {
        const query = params ? '?' + new URLSearchParams(params as Record<string, string>).toString() : '';
        return fetchAPI<{ exercises: Exercise[]; count: number }>(`/exercises${query}`);
    },

    getAllExercises: () =>
        fetchAPI<{ exercises: Exercise[]; count: number }>('/exercises/all'),
};

// ==================== ACHIEVEMENT API ====================

export interface Achievement {
    id: number;
    achievementType: string;
    title: string;
    description: string;
    achievedAt: string;
}

export const achievementAPI = {
    getAchievements: () =>
        fetchAPI<Achievement[]>('/achievements'),

    addAchievement: (data: { achievementType: string; title: string; description?: string }) =>
        fetchAPI<{ message: string; achievement: Achievement }>('/achievements', {
            method: 'POST',
            body: JSON.stringify(data),
        }),

    checkAchievements: () =>
        fetchAPI<{ message: string; newAchievements: Achievement[] }>('/achievements/check', {
            method: 'POST',
        }),
};
