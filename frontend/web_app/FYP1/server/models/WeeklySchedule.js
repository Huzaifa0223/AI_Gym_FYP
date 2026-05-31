const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');

const WeeklySchedule = sequelize.define('WeeklySchedule', {
    id: {
        type: DataTypes.INTEGER,
        primaryKey: true,
        autoIncrement: true
    },
    userId: {
        type: DataTypes.INTEGER,
        allowNull: false,
        field: 'user_id',
        references: {
            model: 'users',
            key: 'id'
        }
    },
    dayOfWeek: {
        type: DataTypes.ENUM('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'),
        allowNull: false,
        field: 'day_of_week'
    },
    muscleGroup: {
        type: DataTypes.STRING(50),
        allowNull: false,
        field: 'muscle_group'
    }
}, {
    tableName: 'weekly_schedules',
    timestamps: true,
    createdAt: 'created_at',
    updatedAt: 'updated_at',
    indexes: [
        {
            unique: true,
            fields: ['user_id', 'day_of_week']
        }
    ]
});

module.exports = WeeklySchedule;
