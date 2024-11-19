#!/home/pi/Downloads/proj_final/camEnv/bin/python

import subprocess
from tkinter import Tk, Button

# Path to the virtual environment's Python interpreter
VENV_PYTHON = "~/Downloads/proj_final/camEnv/bin/python"

def run_bicep_script():
    subprocess.Popen([VENV_PYTHON, "bicep_final.py"])

def run_squat_script():
    subprocess.Popen([VENV_PYTHON, "squat_final.py"])

def run_plank_script():
    subprocess.Popen([VENV_PYTHON, "plank_final.py"])

# Create the main window
app = Tk()
app.title("Exercise Selector")
app.geometry("300x200")

# Create buttons for each script
Button(app, text="Run Bicep Analysis", command=run_bicep_script, height=2, width=20).pack(pady=10)
Button(app, text="Run Squat Analysis", command=run_squat_script, height=2, width=20).pack(pady=10)
Button(app, text="Run Plank Analysis", command=run_plank_script, height=2, width=20).pack(pady=10)

# Run the main loop
app.mainloop()
