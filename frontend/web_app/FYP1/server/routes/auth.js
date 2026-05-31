const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const auth = require('../middleware/auth');

// @route   POST /api/auth/signup
router.post('/signup', authController.signup);

// @route   POST /api/auth/login
router.post('/login', authController.login);

// @route   GET /api/auth/verify
router.get('/verify', auth, authController.verifyToken);

module.exports = router;
