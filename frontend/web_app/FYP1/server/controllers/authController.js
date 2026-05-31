const jwt = require('jsonwebtoken');
const { User, WeeklySchedule } = require('../models');

// Generate JWT token
const generateToken = (userId) => {
    return jwt.sign(
        { userId },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );
};

// Default weekly schedule
const defaultSchedule = {
    Monday: 'Chest',
    Tuesday: 'Biceps',
    Wednesday: 'Back',
    Thursday: 'Triceps',
    Friday: 'Abdominal',
    Saturday: 'Legs',
    Sunday: 'Rest'
};

// @desc    Register new user
// @route   POST /api/auth/signup
// @access  Public
exports.signup = async (req, res) => {
    try {
        const { name, email, password, age, weight, height, goal, experienceMonths, gender, gymTiming } = req.body;

        // Check if user exists
        const existingUser = await User.findOne({ where: { email } });
        if (existingUser) {
            return res.status(400).json({ message: 'User already exists with this email' });
        }

        // Validate required fields
        if (!name || !email || !password || !age || !weight || !height || !goal || !gender) {
            return res.status(400).json({ message: 'Please provide all required fields' });
        }

        // Create user
        const user = await User.create({
            name,
            email,
            password,
            age,
            weight,
            height,
            goal,
            experienceMonths: experienceMonths || 0,
            gender,
            gymTiming: gymTiming || 'morning'
        });

        // Create default weekly schedule for the user
        const scheduleEntries = Object.entries(defaultSchedule).map(([day, muscle]) => ({
            userId: user.id,
            dayOfWeek: day,
            muscleGroup: muscle
        }));

        await WeeklySchedule.bulkCreate(scheduleEntries);

        // Generate token
        const token = generateToken(user.id);

        res.status(201).json({
            message: 'User registered successfully',
            token,
            user: user.toJSON()
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: 'Server error during registration' });
    }
};

// @desc    Login user
// @route   POST /api/auth/login
// @access  Public
exports.login = async (req, res) => {
    try {
        const { email, password } = req.body;

        // Validate input
        if (!email || !password) {
            return res.status(400).json({ message: 'Please provide email and password' });
        }

        // Find user
        const user = await User.findOne({ where: { email } });
        if (!user) {
            return res.status(401).json({ message: 'Invalid email or password' });
        }

        // Compare password
        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(401).json({ message: 'Invalid email or password' });
        }

        // Generate token
        const token = generateToken(user.id);

        res.json({
            message: 'Login successful',
            token,
            user: user.toJSON()
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Server error during login' });
    }
};

// @desc    Verify token and get user
// @route   GET /api/auth/verify
// @access  Private
exports.verifyToken = async (req, res) => {
    try {
        res.json({
            valid: true,
            user: req.user.toJSON()
        });
    } catch (error) {
        console.error('Verify error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
