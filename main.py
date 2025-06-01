
# from util_cam_no_vehicle import real_time_detection
from util_cam import real_time_detection

# For live camera detection
real_time_detection(esp32_cam_ip="10.20.0.215")

# For detecting images in a folder
# detect_images_in_folder("Images")

# For processing a video file
# video_detection("car.mp4")