const express = require('express');
const router = express.Router();
const achievementController = require('../controllers/achievementController');
const auth = require('../middleware/auth');

// All routes require authentication
router.use(auth);

// @route   GET /api/achievements
router.get('/', achievementController.getAchievements);

// @route   POST /api/achievements
router.post('/', achievementController.addAchievement);

// @route   POST /api/achievements/check
router.post('/check', achievementController.checkAchievements);

module.exports = router;
