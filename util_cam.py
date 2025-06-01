import cv2          # OpenCV for computer vision tasks
import re           # Regular expressions for text processing
import easyocr      # OCR (Optical Character Recognition) library
import numpy as np  # Numerical computing
import logging      # Logging system for tracking events
from ultralytics import YOLO  # YOLO object detection model
import time         # Time-related functions
import requests     # HTTP requests (for ESP32-CAM)
from server import save_plate_number  # Custom function to save plates

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader
logger.info("Initializing EasyOCR...")
reader = easyocr.Reader(['en'])

# Load YOLO model (plate detector only)
logger.info("Loading plate detection model...")
vehicle_model = YOLO("weights/yolov11n.pt")
plate_model = YOLO("weights/license_plate_detector.pt")

# Detection parameters
MIN_PLATE_AREA = 200  # Minimum plate area in pixels
PLATE_CONFIDENCE = 0.4  # Minimum confidence for plate detection
EXTRACTION_DELAY = 1.0  # Seconds to wait before text extraction
MIN_SCREEN_COVERAGE = 0.5  # Plate must cover 50% of screen width to extract

# Detection state
detection_state = {
    'first_captured_text': None,
    'current_text': None,
    'detection_start_time': None,
    'last_plate_time': None,
    'plate_visible': False     # NEW: To know if a plate is still on screen
}

def draw_box(image, coords, label=None, color=(0, 255, 0), font_scale=0.6):
    """Enhanced bounding box drawing with better text visibility"""
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    if label:
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), 2)

def preprocess_plate_image(plate_img):
    """Enhanced image preprocessing for better OCR results"""
    try:
        # Upscale image
        plate_img = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(plate_img, -1, kernel)
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=20, templateWindowSize=7, searchWindowSize=21)
        
        # Thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 31, 10)
        return thresh
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def extract_plate_text(plate_img):
    """Robust text extraction with multiple fallback methods"""
    try:
        processed_img = preprocess_plate_image(plate_img)
        if processed_img is None:
            return ""
        
        # Debug: Save processed image
        cv2.imwrite("last_plate_processed.jpg", processed_img)
        
        # Attempt OCR with different parameters
        for attempt in range(3):
            ocr_results = reader.readtext(
                processed_img,
                allowlist='0123456789',
                detail=0,
                text_threshold=0.6 + attempt*0.1,
                width_ths=1.0,
                height_ths=1.0
            )
            
            if ocr_results:
                text = ''.join(ocr_results).replace(' ', '')
                text = re.sub(r'[^0-9]', '', text)
                if len(text) >= 4:  # Minimum plausible plate length
                    logger.info(f"OCR attempt {attempt+1} succeeded: {text}")
                    return text[:8]  # Return first 8 chars max
        
        logger.warning("All OCR attempts failed")
        return ""
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return ""

def detect_plates_only(frame):
    """Direct plate detection without vehicle detection"""
    annotated_frame = frame.copy()
    height, width = frame.shape[:2]
    plate_found = False

    # Detect vehicles first (visualization only)
    vehicle_results = vehicle_model(frame, verbose=False, classes=[2, 3, 5, 7], conf=0.5)
    for vehicle_result in vehicle_results:
        for vehicle_box in vehicle_result.boxes:
            x1, y1, x2, y2 = map(int, vehicle_box.xyxy[0].cpu().numpy())
            draw_box(annotated_frame, [x1, y1, x2, y2], "Vehicle", color=(255, 0, 0))

    # Detect plates directly in the full frame
    plate_results = plate_model(frame, verbose=False, conf=PLATE_CONFIDENCE)

    for plate_result in plate_results:
        for plate_box in plate_result.boxes:
            if plate_box.conf[0] >= PLATE_CONFIDENCE:
                x1, y1, x2, y2 = map(int, plate_box.xyxy[0].cpu().numpy())
                abs_coords = [
                    max(0, x1),
                    max(0, y1),
                    min(width, x2),
                    min(height, y2)
                ]
                plate_width = abs_coords[2] - abs_coords[0]
                plate_height = abs_coords[3] - abs_coords[1]
                area = plate_width * plate_height
                
                # Calculate coverage percentage
                width_coverage = plate_width / width
                
                if area > MIN_PLATE_AREA:
                    plate_found = True
                    detection_state['last_plate_time'] = time.time()
                    
                    # Visual feedback based on coverage
                    if width_coverage >= MIN_SCREEN_COVERAGE:
                        color = (0, 255, 255)  # Yellow - ready for extraction
                        status = "READY"
                    else:
                        color = (0, 0, 255)  # Red - too small
                        status = f"TOO SMALL ({width_coverage*100:.1f}%)"
                    
                    draw_box(annotated_frame, abs_coords, status, color)
                    
                    # Only extract if plate covers enough of screen
                    if width_coverage >= MIN_SCREEN_COVERAGE:
                        if detection_state['current_text'] is None:
                            if detection_state['detection_start_time'] is None:
                                detection_state['detection_start_time'] = time.time()
                                logger.info("Large plate detected, waiting...")
                            elif time.time() - detection_state['detection_start_time'] >= EXTRACTION_DELAY:
                                plate_img = frame[abs_coords[1]:abs_coords[3], abs_coords[0]:abs_coords[2]]
                                plate_text = extract_plate_text(plate_img)
                                
                            if plate_text:
                                detection_state['current_text'] = plate_text
                                detection_state['plate_visible'] = True

                                # Save only if not same as last saved and no visible plate before
                                # if (plate_text != detection_state.get('last_saved_text')) and detection_state.get('last_saved_text') is None:
                                save_plate_number(plate_text)
                                # detection_state['last_saved_text'] = plate_text
                                logger.info(f"Plate saved: {plate_text}")

    # Reset detection if no plates found for 2 seconds
    if not plate_found and detection_state['last_plate_time'] and (time.time() - detection_state['last_plate_time'] > 2):
        logger.info("Plate lost, resetting detection state")
        detection_state.update({
            'current_text': None,
            'detection_start_time': None,
            'plate_visible': False,
            # 'last_saved_text': None  # Now we can save next new plate
        })

    return annotated_frame

def real_time_detection(esp32_cam_ip="192.168.x.x"):
    """Robust ESP32-CAM capture and processing loop"""
    # Reset detection state
    detection_state.update({
        'first_captured_text': None,
        'current_text': None,
        'detection_start_time': None,
        'last_plate_time': None
    })

    capture_url = f"http://{esp32_cam_ip}/capture"
    min_capture_interval = 0.3  # 300ms between captures
    last_capture_time = 0
    frame_count = 0

    logger.info(f"Starting plate-only detection from ESP32-CAM at {esp32_cam_ip}")
    logger.info(f"Minimum coverage: {MIN_SCREEN_COVERAGE*100}% of screen width")
    logger.info("Press 'q' to quit or 's' to save current frame")

    while True:
        try:
            current_time = time.time()
            if current_time - last_capture_time < min_capture_interval:
                continue

            # Capture frame
            response = requests.get(capture_url, timeout=3)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} error")
                time.sleep(1)
                continue

            frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Received empty frame")
                continue

            last_capture_time = current_time
            frame_count += 1

            # Process frame (plate detection only)
            processed_frame = detect_plates_only(frame)

            # Display
            cv2.imshow('Plate Detection Only', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
                logger.info(f"Frame {frame_count} saved for debugging")

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(1)

    cv2.destroyAllWindows()
    logger.info("Detection stopped")

