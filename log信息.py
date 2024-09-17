import logging
import random

logging.basicConfig(filename="Sample.txt",
                    filemode='a',
                    format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Starting the loop for 15 iterations")
for i in range(0,15):
    if(i%2==0):
        logging.warning('Log warning message')
    elif(i%3==0):
        logging.warning('Log critical message')
    else:
        logging.error('Log Error Message')