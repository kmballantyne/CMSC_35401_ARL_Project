import os
import zipfile
# import cv2

# Set paths
zip_path = '/home/kballantyne/CMSC_35401_ARL_Project/CHAOS_DSAL_CT.zip'
extract_to = '/home/kballantyne/CMSC_35401_ARL_Project/CHAOS_DSAL_CT'

# Unzip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Unzipping complete.")