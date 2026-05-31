ALTER USER 'root'@'localhost' IDENTIFIED BY 'root';
UPDATE mysql.user SET authentication_string=null WHERE User='root'; -- Clear potential conflicts
FLUSH PRIVILEGES;
