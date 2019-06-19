# IndoorLocalisation

    
### Configure config file

We have a config file in this project calling ```` config.py ````. This file should be copied and the copied file should 
be renamed to ``` config_dev.py ```. Then you can configure the paths to the origin recorded data.
The constant ``` SAVEPATH ``` defines where the processed data will be saved e.g. the recorded data as numpy array.
The respective folders are created when starting the process/program.
    
### Run project

    python PATH\TO\PROJECT\IndoorLocalisation\main.py
    
### Project structure
For each Process step we have a separated class/file.
The file ``` main.py ``` starts all the Process at once and includes some boolean variables as switches.
    
    # Choose which data file to use
    DATAFILE = 'record1.csv'
    # Visualize the recorded data
    VISUALIZE_RECORDED_DATA = True
    # Apply low pass filter to the norm of the accelerometer
    LOWPASSFILTER = True
    

Set the first switch to True when you want to display the recorded raw data. With the second switch it is possible to
apply a low pass filter to the norm of the acceleration sensor before using it.


When you start it for the first time, some new folders will be created under the path you specified under
```SAVEPATH``` in the configuration file (see the heading [Configure config file](#marker-hHeader-configure-config-file)).
Make sure the recorded data files are stored under the path you specified under ````LOADPATH````


Further classes/files:

```Plotter.py```:  Provides often used functions to plot data

```Reader.py```:  Function to read CSV data at the 'input' path

```AngularIntegration.py```: Calculates the turning angle from the recorded data and estimates the track

```data\recordX.csv```: Recorded raw data

All process-specific settings can be read in the report. 