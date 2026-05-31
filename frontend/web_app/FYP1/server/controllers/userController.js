const bcrypt = require('bcryptjs');
const { User, WeeklySchedule } = require('../models');

// @desc    Get user profile
// @route   GET /api/users/profile
// @access  Private
exports.getProfile = async (req, res) => {
    try {
        const user = await User.findByPk(req.userId, {
            include: [{
                model: WeeklySchedule,
                as: 'weeklySchedule'
            }]
        });

        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        // Transform weekly schedule into object format
        const scheduleObj = {};
        if (user.weeklySchedule) {
            user.weeklySchedule.forEach(schedule => {
                scheduleObj[schedule.dayOfWeek] = schedule.muscleGroup;
            });
        }

        res.json({
            ...user.toJSON(),
            weeklySchedule: scheduleObj
        });
    } catch (error) {
        console.error('Get profile error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Update user profile
// @route   PUT /api/users/profile
// @access  Private
exports.updateProfile = async (req, res) => {
    try {
        const { name, age, weight, height, goal, experienceMonths, gender, gymTiming } = req.body;

        const user = await User.findByPk(req.userId);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        // Update allowed fields
        if (name !== undefined) user.name = name;
        if (age !== undefined) user.age = age;
        if (weight !== undefined) user.weight = weight;
        if (height !== undefined) user.height = height;
        if (goal !== undefined) user.goal = goal;
        if (experienceMonths !== undefined) user.experienceMonths = experienceMonths;
        if (gender !== undefined) user.gender = gender;
        if (gymTiming !== undefined) user.gymTiming = gymTiming;

        await user.save();

        res.json({
            message: 'Profile updated successfully',
            user: user.toJSON()
        });
    } catch (error) {
        console.error('Update profile error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Change user password
// @route   PUT /api/users/change-password
// @access  Private
exports.changePassword = async (req, res) => {
    try {
        const { currentPassword, newPassword } = req.body;

        if (!currentPassword || !newPassword) {
            return res.status(400).json({ message: 'Please provide current and new password' });
        }

        if (newPassword.length < 6) {
            return res.status(400).json({ message: 'New password must be at least 6 characters' });
        }

        const user = await User.findByPk(req.userId);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        // Verify current password
        const isMatch = await user.comparePassword(currentPassword);
        if (!isMatch) {
            return res.status(401).json({ message: 'Current password is incorrect' });
        }

        // Update password (will be hashed by model hook)
        user.password = newPassword;
        await user.save();

        res.json({ message: 'Password changed successfully' });
    } catch (error) {
        console.error('Change password error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Get weekly schedule
// @route   GET /api/users/schedule
// @access  Private
exports.getSchedule = async (req, res) => {
    try {
        const schedules = await WeeklySchedule.findAll({
            where: { userId: req.userId },
            order: [['dayOfWeek', 'ASC']]
        });

        // Transform to object format
        const scheduleObj = {};
        const dayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

        dayOrder.forEach(day => {
            const found = schedules.find(s => s.dayOfWeek === day);
            scheduleObj[day] = found ? found.muscleGroup : 'Rest';
        });

        res.json(scheduleObj);
    } catch (error) {
        console.error('Get schedule error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Update weekly schedule
// @route   PUT /api/users/schedule
// @access  Private
exports.updateSchedule = async (req, res) => {
    try {
        const schedule = req.body;

        if (!schedule || typeof schedule !== 'object') {
            return res.status(400).json({ message: 'Invalid schedule format' });
        }

        const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

        // Update or create schedule entries
        for (const day of days) {
            if (schedule[day]) {
                await WeeklySchedule.upsert({
                    userId: req.userId,
                    dayOfWeek: day,
                    muscleGroup: schedule[day]
                });
            }
        }

        res.json({ message: 'Schedule updated successfully', schedule });
    } catch (error) {
        console.error('Update schedule error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
