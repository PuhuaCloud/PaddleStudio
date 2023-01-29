import os
import shutil

def clear(filepath):
    files = os.listdir(filepath)
    for fd in files:
        cur_path = os.path.join(filepath, fd)            
        if os.path.isdir(cur_path):
            if fd == "__pycache__":
                print("rm %s -rf" % cur_path)
                shutil.rmtree(cur_path)    
            else:
                clear(cur_path)

if __name__ == "__main__":
    clear("./")