@echo off
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" emotiontestenv
python simple-app\src\realtime.py
exit
