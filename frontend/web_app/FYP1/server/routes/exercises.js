const express = require('express');
const router = express.Router();
const exerciseController = require('../controllers/exerciseController');
const auth = require('../middleware/auth');

// All routes require authentication
router.use(auth);

// @route   GET /api/exercises
router.get('/', exerciseController.getExercises);

// @route   GET /api/exercises/all
router.get('/all', exerciseController.getAllExercises);

module.exports = router;
