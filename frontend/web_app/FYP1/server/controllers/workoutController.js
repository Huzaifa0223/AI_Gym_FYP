const { WorkoutHistory } = require('../models');
const { Op } = require('sequelize');

// @desc    Get workout history
// @route   GET /api/workouts
// @access  Private
exports.getWorkouts = async (req, res) => {
    try {
        const { limit = 20, offset = 0, startDate, endDate } = req.query;

        const whereClause = { userId: req.userId };

        // Date filtering
        if (startDate || endDate) {
            whereClause.workoutDate = {};
            if (startDate) whereClause.workoutDate[Op.gte] = startDate;
            if (endDate) whereClause.workoutDate[Op.lte] = endDate;
        }

        const workouts = await WorkoutHistory.findAndCountAll({
            where: whereClause,
            order: [['workoutDate', 'DESC'], ['created_at', 'DESC']],
            limit: parseInt(limit),
            offset: parseInt(offset)
        });

        res.json({
            workouts: workouts.rows,
            total: workouts.count,
            limit: parseInt(limit),
            offset: parseInt(offset)
        });
    } catch (error) {
        console.error('Get workouts error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Get workout stats
// @route   GET /api/workouts/stats
// @access  Private
exports.getStats = async (req, res) => {
    try {
        const userId = req.userId;

        // Get all workouts for this user
        const allWorkouts = await WorkoutHistory.findAll({
            where: { userId }
        });

        // Get workouts from this week
        const today = new Date();
        const weekStart = new Date(today);
        weekStart.setDate(today.getDate() - today.getDay());
        weekStart.setHours(0, 0, 0, 0);

        const weeklyWorkouts = allWorkouts.filter(w =>
            new Date(w.workoutDate) >= weekStart
        );

        // Calculate stats
        const totalWorkouts = allWorkouts.length;
        const weeklyWorkoutsCount = weeklyWorkouts.length;

        const totalReps = allWorkouts.reduce((sum, w) => sum + (w.reps || 0), 0);
        const totalCalories = allWorkouts.reduce((sum, w) => sum + (w.caloriesBurned || 0), 0);

        const accuracies = allWorkouts.filter(w => w.accuracy).map(w => parseFloat(w.accuracy));
        const averageAccuracy = accuracies.length > 0
            ? Math.round(accuracies.reduce((a, b) => a + b, 0) / accuracies.length)
            : 0;

        res.json({
            totalWorkouts,
            weeklyWorkouts: weeklyWorkoutsCount,
            averageAccuracy,
            caloriesBurned: totalCalories,
            totalReps,
            totalDuration: allWorkouts.reduce((sum, w) => sum + (w.durationMinutes || 0), 0)
        });
    } catch (error) {
        console.error('Get stats error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Add new workout
// @route   POST /api/workouts
// @access  Private
exports.addWorkout = async (req, res) => {
    try {
        const {
            exerciseName,
            muscleGroup,
            reps,
            sets,
            accuracy,
            durationMinutes,
            caloriesBurned,
            workoutDate
        } = req.body;

        if (!exerciseName || !muscleGroup) {
            return res.status(400).json({ message: 'Exercise name and muscle group are required' });
        }

        const workout = await WorkoutHistory.create({
            userId: req.userId,
            exerciseName,
            muscleGroup,
            reps: reps || 0,
            sets: sets || 0,
            accuracy: accuracy || null,
            durationMinutes: durationMinutes || null,
            caloriesBurned: caloriesBurned || null,
            workoutDate: workoutDate || new Date().toISOString().split('T')[0]
        });

        res.status(201).json({
            message: 'Workout logged successfully',
            workout
        });
    } catch (error) {
        console.error('Add workout error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Get single workout
// @route   GET /api/workouts/:id
// @access  Private
exports.getWorkout = async (req, res) => {
    try {
        const workout = await WorkoutHistory.findOne({
            where: {
                id: req.params.id,
                userId: req.userId
            }
        });

        if (!workout) {
            return res.status(404).json({ message: 'Workout not found' });
        }

        res.json(workout);
    } catch (error) {
        console.error('Get workout error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Delete workout
// @route   DELETE /api/workouts/:id
// @access  Private
exports.deleteWorkout = async (req, res) => {
    try {
        const result = await WorkoutHistory.destroy({
            where: {
                id: req.params.id,
                userId: req.userId
            }
        });

        if (!result) {
            return res.status(404).json({ message: 'Workout not found' });
        }

        res.json({ message: 'Workout deleted successfully' });
    } catch (error) {
        console.error('Delete workout error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
