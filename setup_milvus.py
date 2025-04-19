import os
import subprocess
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_docker_running():
    """Check if Docker is running."""
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_milvus_running():
    """Check if Milvus container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return "milvus-standalone" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_milvus():
    """Start Milvus using docker-compose."""
    try:
        if not check_docker_running():
            logger.error("Docker is not running. Please start Docker and try again.")
            return False
            
        logger.info("Starting Milvus with docker-compose...")
        subprocess.run(
            ["docker-compose", "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Wait for Milvus to be ready
        max_attempts = 15
        for attempt in range(max_attempts):
            if check_milvus_running():
                logger.info(f"Milvus is now running (attempt {attempt + 1}/{max_attempts})")
                # Give it a bit more time to fully initialize
                time.sleep(5)
                return True
            else:
                logger.info(f"Waiting for Milvus to start... ({attempt + 1}/{max_attempts})")
                time.sleep(10)
        
        logger.error("Milvus failed to start within the expected time")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting Milvus: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False

def setup_milvus():
    """Check if Milvus is running, and start it if not."""
    if check_milvus_running():
        logger.info("Milvus is already running")
        return True
    else:
        logger.info("Milvus is not running, attempting to start it")
        return start_milvus()

if __name__ == "__main__":
    if setup_milvus():
        logger.info("Milvus setup completed successfully")
    else:
        logger.error("Failed to set up Milvus") 