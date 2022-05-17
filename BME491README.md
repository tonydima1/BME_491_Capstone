# BME_491_Capstone:
Capstone project from UNR. A sleep apnea device able to predict if user has OSA (obstructive sleep apnea).

Arduino_Sensor_Collect.ino: 
Uses the max30102 sensor to connect collect IR light and Red light and send it to the ardunio's serial output.

Collect.py:
Collects the live data from the arduino serial output. Done on a raspberry pi as the arduino lacks RAM memory capabilities to support a full data set. 
Saves it as an editable .csv

Analysis.py:
Takes the .csv file and reads it into various arrays allowing for complex data analysis as needed. Automatically runs at the end of the Collect.py. 
It is separated as to allow for differing data sets from outside sources or to rerun a saved data set from a previous collection. 
This program presents the data to the user. Heart rate, Spo2, OSA prediction, filtered, and unfiltered data. It saves data accordingly.
