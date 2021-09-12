# This function is for quickly cleaning up cache directories
import os
import shutil


def main():
    dirpath = os.getcwd()

    dirs = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirpath):
        for dirname in d:
            if '__pycache__' in dirname:
                dirs.append(os.path.join(r, dirname))
    #print('Cleaning caches...')
    #print(dirs)
    for dirname in dirs:
        shutil.rmtree(dirname)


main()
