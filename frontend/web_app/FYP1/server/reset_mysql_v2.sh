#!/bin/bash
echo "STOPPING..."
brew services stop mysql
pkill mysqld
sleep 5

echo "RESETTING..."
# Run mysqld with the init file to reset password
mysqld --default-authentication-plugin=mysql_native_password --init-file=/Users/mac/Desktop/figma-local/server/mysql-init.sql &
PID=$!
echo "Running PID: $PID"
sleep 15
kill $PID
sleep 5

echo "STARTING..."
brew services start mysql
sleep 5

echo "TESTING..."
mysql -u root -proot -e "SELECT 'SUCCESS' as status;"
