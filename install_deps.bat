@echo off

:: Install pip (https://bootstrap.pypa.io/get-pip.py)
C:\Python27\python.exe get-pip.py

:: Install deps
C:\Python27\Scripts\pip.exe install Pillow

pause