import psycopg2
import logging
from psycopg2 import sql
from serial import Serial
import time

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'dbname': 'ANPR',
    'user': 'postgres',
    'password': '1233210002281Sliman',
    'host': 'localhost',
    'port': 5432,
}

# Serial configuration
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
MOTOR_OPEN_TIME_SECONDS = 5

def control_motor(open_time=MOTOR_OPEN_TIME_SECONDS):
    try:
        logger.info("Opening serial port to control motor...")
        with Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as port:
            time.sleep(2)  # Wait for Arduino reset
            port.write(b'O')
            port.flush()
            logger.info("Motor opened (command 'O' sent).")
            time.sleep(open_time)
            port.write(b'C')
            port.flush()
            logger.info("Motor closed (command 'C' sent).")
    except Exception as e:
        logger.error(f"Failed to control motor via COM5: {e}")

def save_plate_number(plate_number):
    """
    Save the plate number if it's new, or trigger the motor if it already exists.
    Always closes the DB connection.
    """
    conn = None
    try:
        if not plate_number or plate_number.strip().lower() == "no text":
            logger.warning("Invalid or empty plate number. Skipping.")
            return

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check for duplicates in the last 5 minutes
        cursor.execute("""
            SELECT id FROM Detection_Table
            WHERE PlateNumber = %s AND allow = true;
        """, (plate_number,))
        existing = cursor.fetchone()

        if existing:
            logger.info(f"This vehicle is allowed: {plate_number} → Opening gate.")
            control_motor()
        else:
            # Insert new record
            cursor.execute(sql.SQL("""
            INSERT INTO Detection_Table (PlateNumber, datetime, allow)
            SELECT %s, NOW(), false
            WHERE NOT EXISTS (
                SELECT 1 FROM Detection_Table WHERE PlateNumber = %s
            );
            """), (plate_number,plate_number))
            conn.commit()
            logger.info(f"New plate saved: {plate_number}")

    except Exception as e:
        logger.error(f"Database error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            logger.info("Database connection closed.")
