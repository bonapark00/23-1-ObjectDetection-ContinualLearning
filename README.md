# Instructions for running the code
# Using Python virtualenv

## 1. Setting up and activating the virtual environment
  
  #### a) For Windows: 
  ```
    py -m pip install --user virtualenv 
  ```
  This downloads the packages necessary for creating virtual environments on your system.
  ```  
    py -m venv venv 
  ```
  This creates a virtual environment.
  ```
    venv\Scripts\activate.bat
  ```
  This activates your virtual environment. Working on the virtual environment ensures the stability of dependencies used for the project.
  
  #### b) For Linux/MacOS:
  ```
     python3 -m pip install --user virtualenv
  ```
  ```
     python3 -m venv venv
  ```
  ```
     source venv/bin/activate
  ```

You may now navigate to the directory containing your backend repository using the terminal. Now you can run the following command:
  
  #### a) For Windows/Linux/MacOS:
  ```
    pip install -r requirements.txt
  ```
  ```
    pip freeze > requirements.txt
  ```
  
This installs all the requirements in the requirements.txt file and freezes them to prevent updates.
If you wish, you can install all the required packages step-by-step as well.

## 2. Using modified torchvision

Run the script in ```/scripts/update_torchvision``` using dir2 for virtualenv( use dir for docker environments)
