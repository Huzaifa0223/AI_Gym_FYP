const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');

const Achievement = sequelize.define('Achievement', {
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
    achievementType: {
        type: DataTypes.STRING(50),
        allowNull: false,
        field: 'achievement_type'
    },
    title: {
        type: DataTypes.STRING(100),
        allowNull: false
    },
    description: {
        type: DataTypes.TEXT,
        allowNull: true
    },
    achievedAt: {
        type: DataTypes.DATE,
        defaultValue: DataTypes.NOW,
        field: 'achieved_at'
    }
}, {
    tableName: 'achievements',
    timestamps: false,
    indexes: [
        {
            fields: ['user_id']
        }
    ]
});

module.exports = Achievement;
