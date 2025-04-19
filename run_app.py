import os
import subprocess
import logging
from setup_milvus import setup_milvus

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_streamlit_app():
    """Run the Streamlit application."""
    try:
        # Check if .env file exists
        if not os.path.exists(".env"):
            logger.warning(".env file not found. Creating example .env file...")
            with open(".env", "w") as f:
                f.write("OPENAI_API_KEY=your_openai_api_key\n")
                f.write("MILVUS_HOST=localhost\n")
                f.write("MILVUS_PORT=19530\n")
            logger.info("Created .env file. Please edit it to add your OpenAI API key.")

        # Ensure Milvus is running
        if not setup_milvus():
            logger.error("Failed to setup Milvus. Streamlit app may not function correctly.")
            return False

        logger.info("Starting Streamlit app...")
        subprocess.run(["streamlit", "run", "main.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return True

if __name__ == "__main__":
    print("""
    #############################################
    #                                           #
    #   Digital Twin Email Response Generator   #
    #                                           #
    #############################################
    """)
    
    print("Starting application...")
    run_streamlit_app() 