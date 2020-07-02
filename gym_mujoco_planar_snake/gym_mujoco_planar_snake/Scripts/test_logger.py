from baselines import logger

logger.configure(dir="~/Desktop/")
logger.log("Hello")

logger.logkv("a", 4)