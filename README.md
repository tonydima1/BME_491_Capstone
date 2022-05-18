# BME_491_Capstone:
Capstone project from UNR. A sleep apnea device able to predict if user has OSA (obstructive sleep apnea).

Arduino_Sensor_Collect.ino: 
Uses the max30102 sensor to connect collect IR light and Red light and send it to the ardunio's serial output. 
- Setup the sensor. 
- Detect sensor connection.
- Set settings for sensor indicated by library.
- Loop for data capture. 
- Make data avalible to other softwares.
- Print on serial monitor.

Collect.py:
Collects the live data from the arduino serial output. Done on a raspberry pi as the arduino lacks RAM memory capabilities to support a full data set. 
Saves it as an editable .csv
- Initialize serial port.
- Set up file for collection.
- Remove old file (If same name).
- Collect from arduino until interrupted.
- Filter out data (parameters can be changed) and fix time vector.
- Save data.
- Run Analysis.py.

Analysis.py:
Takes the .csv file and reads it into various arrays allowing for complex data analysis as needed. Automatically runs at the end of the Collect.py. 
It is separated as to allow for differing data sets from outside sources or to rerun a saved data set from a previous collection. 
This program presents the data to the user. Heart rate, Spo2, OSA prediction, filtered, and unfiltered data. It saves data accordingly.
- Set up data arrays from file.
- Determine ranges from sample rate and heart rate.
- Plot raw data.
- Plot filtered data with peaks placed, and heart rate displayed for Red light.
- Plot filtered data with peaks placed, and heart rate displayed for IR light.
- Calculate Spo2.
- Filter Spo2 values.
- Detect peaks in Spo2 data. (Keeping it relative to time).
- Display data to user Heart rate, Spo2, averages, and plot.
- Plot the peaks on the Spo2 plot.
- Detect drop instances.
- Make OSA prediction.
