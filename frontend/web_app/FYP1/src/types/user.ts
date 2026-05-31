export interface User {
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
