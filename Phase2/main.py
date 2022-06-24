from pathlib import Path
from logzero import logger, logfile
from sense_hat import SenseHat
from datetime import datetime, timedelta
from orbit import ISS
import numpy as np
import csv
import os
import math
from picamera import PiCamera
from skyfield.api import load
import cv2 as cv
from fastiecm import fastiecm


base_folder = Path(__file__).parent.resolve()

# Set a logfile name
logfile(base_folder/"events.log")

# Set-up sense
sense = SenseHat()

# Set up camera
cam = PiCamera()
cam.resolution = (1920, 1080)


def create_csv_file(data_file):
    """Create a new CSV file and add the header row"""
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        header = ('Row_Id','Timestamp','Latitude','Longitude','Mag_x','Mag_y','Mag_z','Magnetic_grad','Accel_x','Accel_y','Accel_z','Pitch','Roll','Yaw')
        writer.writerow(header)


def add_csv_data(data_file, data):
    """Add a row of data to the data_file CSV"""
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def convert(angle):
    """
    Convert a `skyfield` Angle to an EXIF-appropriate
    representation (rationals). Return a tuple containing a 
    boolean and the converted angle, with the boolean 
    indicating if the angle is negative.
    """
    sign, degrees, minutes, seconds = angle.signed_dms()
    exif_angle = f'{degrees:.0f}/1,{minutes:.0f}/1,{seconds*10:.0f}/10'
    return sign < 0, exif_angle

def capture(camera, image):
    """Use `camera` to capture an `image` file with lat/long EXIF data."""
    location = ISS.coordinates()

    # Convert the latitude and longitude to EXIF-appropriate representations
    south, exif_latitude = convert(location.latitude)
    west, exif_longitude = convert(location.longitude)

    # Set the EXIF tags specifying the current location
    camera.exif_tags['GPS.GPSLatitude'] = exif_latitude
    camera.exif_tags['GPS.GPSLatitudeRef'] = "S" if south else "N"
    camera.exif_tags['GPS.GPSLongitude'] = exif_longitude
    camera.exif_tags['GPS.GPSLongitudeRef'] = "W" if west else "E"

    # Capture the image
    camera.capture(image)

def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calc_ndvi(im):
    b, g, r = cv.split(im)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom == 0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

# Main program
try:
    # load ephemeris for the ISS position
    ephemeris = load('/home/pi/de421.bsp')
    timescale = load.timescale()

    # Intialize data counter and start and current time
    count = 0
    phcounter = 0
    start_time = datetime.now()
    now_time = datetime.now()

    # Create image folder
    if (not os.path.exists(str(base_folder) + '/log')):
        os.mkdir(str(base_folder) + '/log')
    if (not os.path.exists(str(base_folder) + '/pic')):
        os.mkdir(str(base_folder) + '/pic')
    filetime = datetime.now().strftime("%d_%b_%y_%H_%M_%S")      # Time stamp variable to be written in the log file
    filename = str(base_folder) + '/log/'+ (str(filetime)) + '_recording_log.csv'      # Recored log will be stored in a log folder
    create_csv_file(filename)

    print("Logging data started at " + datetime.now().strftime("%d %b %y %H:%M:%S"))

    # Start the recording cycle
    while (now_time < start_time + timedelta(minutes = 175)):

        timestmp = datetime.now()
        
        # Get coordinates of location on Earth below the ISS
        location = ISS.coordinates()    

        # Magneometer readings
        sense.set_imu_config(True, False, False)
        rawComp  = sense.get_compass_raw()
        rawCompX = round(rawComp['x'], 3)
        rawCompY = round(rawComp['y'], 3)
        rawCompZ = round(rawComp['z'], 3)
 
        # The gradient of the measured magnetic field
        gradient = round(math.sqrt(rawCompX**2 + rawCompY**2 + rawCompZ**2), 3)  

        # Acceleration readings
        sense.set_imu_config(False, False, True)
        rawAcc  = sense.get_accelerometer_raw()
        rawAccX = round(rawAcc['x'], 3)
        rawAccY = round(rawAcc['y'], 3)
        rawAccZ = round(rawAcc['z'], 3)

        # Orientation readings
        sense.set_imu_config(False, True, True)
        orient = sense.get_orientation()
        pitch  = round(orient['pitch'], 2)
        roll   = round(orient['roll'], 2)
        yaw    = round(orient['yaw'], 2)

        # Gather data toghether for logging
        data = (count, str(timestmp), round(location.latitude.degrees, 4), round(location.longitude.degrees,4), 
                rawCompX, rawCompY, rawCompZ, gradient, 
                rawAccX, rawAccY, rawAccZ,
                pitch, roll, yaw)
        print(data)
        add_csv_data(filename, data)

        # Take a photo
        if ISS.at(timescale.now()).is_sunlit(ephemeris):
            if count % 200 == 0:
                image_file = f"{base_folder}/pic/photo_{phcounter:03d}.jpg"
                capture(cam, image_file)
                phcounter += 1
                logger.info('Photo taken at {}'.format(str(timestmp)))
                image = cv.imread(image_file) # load image
                image = np.array(image, dtype=float)/float(255) #convert to an array
                contrasted = contrast_stretch(image)
                cv.imwrite(image_file, contrasted)
                ndvi = calc_ndvi(contrasted)
                ndvi_contrasted = contrast_stretch(ndvi)
                color_mapped_prep = ndvi_contrasted.astype(np.uint8)
                color_mapped_image = cv.applyColorMap(color_mapped_prep, fastiecm)
                cv.imwrite(image_file[:-3]+'ndvi.png', color_mapped_image)

        count += 1
        now_time = datetime.now()
       
except Exception as e:
    logger.error(f'{e.__class__.__name__}: {e}')

finally:    # Exception handler
    if (count > 0):
        print("Recorded measurements: " + str(count-1))
    else:
        print("No data recorded")
    print('Thank you')

