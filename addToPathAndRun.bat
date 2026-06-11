@echo off
rem Exécuter la commande PowerShell et rediriger uniquement le chemin vers un fichier
powershell -NoLogo -NoProfile -Command "(Resolve-Path .\build\subprojects\glfw-3.3.10).Path" > temp.txt

rem Lire la première ligne du fichier dans une variable
set /p pathToDLL=<temp.txt

rem Afficher le contenu de la variable
set path=%pathToDLL%;%path%

rem Nettoyer le fichier de sortie
del temp.txt

rem Lancer le programme
build\main.exe