const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');

const WorkoutHistory = sequelize.define('WorkoutHistory', {
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
    exerciseName: {
        type: DataTypes.STRING(100),
        allowNull: false,
        field: 'exercise_name'
    },
    muscleGroup: {
        type: DataTypes.STRING(50),
        allowNull: false,
        field: 'muscle_group'
    },
    reps: {
        type: DataTypes.INTEGER,
        allowNull: false,
        defaultValue: 0
    },
    sets: {
        type: DataTypes.INTEGER,
        allowNull: false,
        defaultValue: 0
    },
    accuracy: {
        type: DataTypes.DECIMAL(5, 2),
        allowNull: true
    },
    durationMinutes: {
        type: DataTypes.INTEGER,
        allowNull: true,
        field: 'duration_minutes'
    },
    caloriesBurned: {
        type: DataTypes.INTEGER,
        allowNull: true,
        field: 'calories_burned'
    },
    workoutDate: {
        type: DataTypes.DATEONLY,
        allowNull: false,
        field: 'workout_date'
    }
}, {
    tableName: 'workout_history',
    timestamps: true,
    createdAt: 'created_at',
    updatedAt: false,
    indexes: [
        {
            fields: ['user_id', 'workout_date']
        }
    ]
});

module.exports = WorkoutHistory;
