const express = require('express');
const router = express.Router();
const workoutController = require('../controllers/workoutController');
const auth = require('../middleware/auth');

// All routes require authentication
router.use(auth);

// @route   GET /api/workouts
router.get('/', workoutController.getWorkouts);

// @route   GET /api/workouts/stats
router.get('/stats', workoutController.getStats);

// @route   POST /api/workouts
router.post('/', workoutController.addWorkout);

// @route   GET /api/workouts/:id
router.get('/:id', workoutController.getWorkout);

// @route   DELETE /api/workouts/:id
router.delete('/:id', workoutController.deleteWorkout);

module.exports = router;
