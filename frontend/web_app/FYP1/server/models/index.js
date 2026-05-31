const sequelize = require('../config/database');
const User = require('./User');
const WeeklySchedule = require('./WeeklySchedule');
const WorkoutHistory = require('./WorkoutHistory');
const Achievement = require('./Achievement');

// Define associations
User.hasMany(WeeklySchedule, { foreignKey: 'userId', as: 'weeklySchedule' });
WeeklySchedule.belongsTo(User, { foreignKey: 'userId' });

User.hasMany(WorkoutHistory, { foreignKey: 'userId', as: 'workoutHistory' });
WorkoutHistory.belongsTo(User, { foreignKey: 'userId' });

User.hasMany(Achievement, { foreignKey: 'userId', as: 'achievements' });
Achievement.belongsTo(User, { foreignKey: 'userId' });

module.exports = {
    sequelize,
    User,
    WeeklySchedule,
    WorkoutHistory,
    Achievement
};
