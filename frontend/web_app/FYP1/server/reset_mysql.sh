#!/bin/bash
echo "🛑 Stopping MySQL..."
brew services stop mysql

echo "🔓 Starting in safe mode..."
# Kill any rogue processes
pkill mysqld
mysqld_safe --skip-grant-tables > /dev/null 2>&1 &
PID=$!
sleep 5

echo "🔑 Setting password to 'root'..."
mysql -u root <<EOF
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY 'root';
FLUSH PRIVILEGES;
EOF

echo "🛑 Stopping safe mode..."
kill $PID
sleep 3
pkill mysqld

echo "🚀 Restarting MySQL..."
brew services start mysql
sleep 5

echo "✅ Testing connection..."
mysql -u root -proot -e "SELECT 'SUCCESS: Password set to root' as status;"
