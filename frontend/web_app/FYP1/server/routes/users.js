const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const auth = require('../middleware/auth');

// All routes require authentication
router.use(auth);

// @route   GET /api/users/profile
router.get('/profile', userController.getProfile);

// @route   PUT /api/users/profile
router.put('/profile', userController.updateProfile);

// @route   PUT /api/users/change-password
router.put('/change-password', userController.changePassword);

// @route   GET /api/users/schedule
router.get('/schedule', userController.getSchedule);

// @route   PUT /api/users/schedule
router.put('/schedule', userController.updateSchedule);

module.exports = router;
