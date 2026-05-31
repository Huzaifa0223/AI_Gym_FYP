const { Achievement } = require('../models');

// @desc    Get user achievements
// @route   GET /api/achievements
// @access  Private
exports.getAchievements = async (req, res) => {
    try {
        const achievements = await Achievement.findAll({
            where: { userId: req.userId },
            order: [['achievedAt', 'DESC']]
        });

        res.json(achievements);
    } catch (error) {
        console.error('Get achievements error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Add achievement
// @route   POST /api/achievements
// @access  Private
exports.addAchievement = async (req, res) => {
    try {
        const { achievementType, title, description } = req.body;

        if (!achievementType || !title) {
            return res.status(400).json({ message: 'Achievement type and title are required' });
        }

        // Check if this achievement type already exists for user
        const existing = await Achievement.findOne({
            where: {
                userId: req.userId,
                achievementType
            }
        });

        if (existing) {
            return res.status(400).json({ message: 'Achievement already earned' });
        }

        const achievement = await Achievement.create({
            userId: req.userId,
            achievementType,
            title,
            description
        });

        res.status(201).json({
            message: 'Achievement unlocked!',
            achievement
        });
    } catch (error) {
        console.error('Add achievement error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// @desc    Check and award achievements
// @route   POST /api/achievements/check
// @access  Private
exports.checkAchievements = async (req, res) => {
    try {
        const { WorkoutHistory } = require('../models');
        const userId = req.userId;
        const newAchievements = [];

        // Get user's workout data
        const workouts = await WorkoutHistory.findAll({ where: { userId } });
        const existingAchievements = await Achievement.findAll({ where: { userId } });
        const earnedTypes = existingAchievements.map(a => a.achievementType);

        // Achievement checks
        const checks = [
            {
                type: 'first_workout',
                title: 'First Steps',
                description: 'Completed your first workout!',
                condition: workouts.length >= 1
            },
            {
                type: 'workout_10',
                title: 'Getting Started',
                description: 'Completed 10 workouts!',
                condition: workouts.length >= 10
            },
            {
                type: 'workout_50',
                title: 'Dedicated',
                description: 'Completed 50 workouts!',
                condition: workouts.length >= 50
            },
            {
                type: 'workout_100',
                title: 'Centurion',
                description: 'Completed 100 workouts!',
                condition: workouts.length >= 100
            },
            {
                type: 'total_100_reps',
                title: 'Rep Counter',
                description: 'Completed 100 total reps!',
                condition: workouts.reduce((sum, w) => sum + (w.reps || 0), 0) >= 100
            },
            {
                type: 'total_1000_reps',
                title: 'Rep Master',
                description: 'Completed 1000 total reps!',
                condition: workouts.reduce((sum, w) => sum + (w.reps || 0), 0) >= 1000
            },
            {
                type: 'accuracy_90',
                title: 'Perfect Form',
                description: 'Achieved 90%+ accuracy on a workout!',
                condition: workouts.some(w => parseFloat(w.accuracy) >= 90)
            },
            {
                type: 'accuracy_95',
                title: 'Form Master',
                description: 'Achieved 95%+ accuracy on a workout!',
                condition: workouts.some(w => parseFloat(w.accuracy) >= 95)
            }
        ];

        // Award new achievements
        for (const check of checks) {
            if (check.condition && !earnedTypes.includes(check.type)) {
                const achievement = await Achievement.create({
                    userId,
                    achievementType: check.type,
                    title: check.title,
                    description: check.description
                });
                newAchievements.push(achievement);
            }
        }

        res.json({
            message: newAchievements.length > 0
                ? `${newAchievements.length} new achievement(s) unlocked!`
                : 'No new achievements',
            newAchievements
        });
    } catch (error) {
        console.error('Check achievements error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};
