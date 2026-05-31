-- AI Gym Trainer Database Schema
-- Run this script to create the database and tables

CREATE DATABASE IF NOT EXISTS ai_gym_trainer;
USE ai_gym_trainer;

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password VARCHAR(255) NOT NULL,
  age INT NOT NULL,
  weight DECIMAL(5,2) NOT NULL,
  height DECIMAL(5,2) NOT NULL,
  goal ENUM('gain', 'loss', 'maintenance') NOT NULL DEFAULT 'maintenance',
  experience_months INT DEFAULT 0,
  gender ENUM('male', 'female', 'other') NOT NULL DEFAULT 'male',
  gym_timing VARCHAR(50) DEFAULT 'morning',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Weekly Schedule table
CREATE TABLE IF NOT EXISTS weekly_schedules (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  day_of_week ENUM('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday') NOT NULL,
  muscle_group VARCHAR(50) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  UNIQUE KEY unique_user_day (user_id, day_of_week)
);

-- Workout History table
CREATE TABLE IF NOT EXISTS workout_history (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  exercise_name VARCHAR(100) NOT NULL,
  muscle_group VARCHAR(50) NOT NULL,
  reps INT NOT NULL DEFAULT 0,
  sets INT NOT NULL DEFAULT 0,
  accuracy DECIMAL(5,2) DEFAULT NULL,
  duration_minutes INT DEFAULT NULL,
  calories_burned INT DEFAULT NULL,
  workout_date DATE NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_user_date (user_id, workout_date)
);

-- Achievements table
CREATE TABLE IF NOT EXISTS achievements (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  achievement_type VARCHAR(50) NOT NULL,
  title VARCHAR(100) NOT NULL,
  description TEXT,
  achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_user_achievements (user_id)
);

-- Insert default achievements types for reference
-- These can be earned by users:
-- 'first_workout' - Complete first workout
-- 'streak_7' - 7-day workout streak
-- 'streak_30' - 30-day workout streak
-- 'total_100_reps' - Complete 100 total reps
-- 'total_1000_reps' - Complete 1000 total reps
-- 'accuracy_90' - Achieve 90%+ accuracy on a workout
-- 'muscle_master' - Complete all muscle groups in a week
