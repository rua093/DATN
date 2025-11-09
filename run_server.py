"""Run the server with scheduler"""
import uvicorn
from server.main import app
from server.scheduler import start_scheduler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Start scheduler
    logger.info("Starting scheduler...")
    start_scheduler()
    
    # Start FastAPI server
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    logger.info("API documentation available at http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
