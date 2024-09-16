import time
import logging  # 引入logging模块
from pathlib import Path

'''得到项目的根目录'''
def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)
 
class Logger:
    def __init__(self,mode='a'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%m-%d_%H-%M', time.localtime(time.time()))
        root_path = get_project_root()
        log_path = root_path + '/log/'
        log_name = f'{log_path}{rq}.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)  #日志文件追加在末尾 
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
if __name__=='__main__':
    logger = Logger()
     # 写入日志
    logger.logger.info('这是一条测试日志模块能否正常运行的记录......\n')