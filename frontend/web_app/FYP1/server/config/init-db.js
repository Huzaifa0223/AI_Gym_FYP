require('dotenv').config();
const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');

async function initDatabase() {
  console.log('🔧 Initializing MySQL database...');

  try {
    // Connect without specifying a database first to create it
    const connection = await mysql.createConnection({
      host: process.env.DB_HOST || 'localhost',
      port: process.env.DB_PORT || 3307,
      user: process.env.DB_USER || 'root',
      password: process.env.DB_PASSWORD || '',
      multipleStatements: true
    });

    console.log('✅ Connected to MySQL server');

    // Read and execute schema
    const schemaPath = path.join(__dirname, '..', 'schema.sql');
    const schema = fs.readFileSync(schemaPath, 'utf8');

    console.log('📄 Executing schema...');
    await connection.query(schema);

    console.log('✅ Database and tables created successfully!');
    console.log('');
    console.log('📊 Created tables:');
    console.log('   - users');
    console.log('   - weekly_schedules');
    console.log('   - workout_history');
    console.log('   - achievements');
    console.log('');
    console.log('🚀 You can now start the server with: npm run dev');

    await connection.end();
    process.exit(0);
  } catch (error) {
    console.error('❌ Error initializing database:', error.message);
    console.error('');
    console.error('Make sure:');
    console.error('1. MySQL is running');
    console.error('2. Update the .env file with correct credentials');
    console.error('');
    process.exit(1);
  }
}

initDatabase();
