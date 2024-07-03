import inspect
import os

def debug(*args, **kwargs):
    frame_info = inspect.getframeinfo(inspect.currentframe().f_back)
    path = frame_info.filename
    filename = os.path.basename(path)
    lineno = frame_info.lineno

    print(f"[DEBUG][{filename}:{lineno}]", *args, **kwargs)
