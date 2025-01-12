# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 05/01/25
import logging
import datetime
def update_version_file():
    """
    Reads the version from 'version.txt', increments it by 1,
    and overwrites the file with the new version.

    Assumes 'version.txt' contains a single line with the version number.
    """
    try:
        new_version = 0
        with open('__version.txt', 'r') as f:
            current_version = int(f.read().strip())

        new_version = current_version + 1



    except FileNotFoundError:
        print("Error: 'version.txt' not found.")
    except ValueError:
        print("Error: Invalid version number in 'version.txt'.")
    with open('version.txt', 'w') as f:
        f.write(str(new_version))

    print(f"Version updated to: {new_version}")
    return new_version

new_version = update_version_file()
# Define the log file path and name
log_filename = f"log/run/v_{new_version}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create file handler to write logs to a file
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Create console handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define log message format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)