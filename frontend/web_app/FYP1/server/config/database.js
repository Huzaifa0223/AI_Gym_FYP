require('dotenv').config();
const { Sequelize } = require('sequelize');
const path = require('path');

const sequelize = new Sequelize({
    dialect: 'sqlite',
    storage: path.join(__dirname, '..', 'ai_gym_trainer.sqlite'),
    logging: process.env.NODE_ENV === 'development' ? console.log : false,
});

module.exports = sequelize;
