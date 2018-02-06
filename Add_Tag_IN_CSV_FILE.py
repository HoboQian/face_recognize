import sys
import os
import glob

# -*- coding: utf-8 -*-
author__ = 'hobo'

FILE_PATH = 'F:/Work_File_share/Face_Recognize_hobo/raw/'
FILE_NAME = 'train_data.txt'
GENERATE_FILE_NAME = 'csv.txt'
COMPLETE_RAW_FILE_PATH = FILE_PATH + FILE_NAME
COMPLETE_GENERATE_FILE_PATH = FILE_PATH + GENERATE_FILE_NAME

def add_tag_in_csv_file():
    raw_file = open(COMPLETE_RAW_FILE_PATH, 'r')
    generate_file = open(COMPLETE_GENERATE_FILE_PATH, 'a+')

    if raw_file is None:
        print ('ERROR: RAW file can not been opened.')
        return

    NAME = ''                                              # used to distinguish name
    TAG_NUM = -1
    for line in raw_file:
        line1 = line.replace('\\', '/')
        line2 = line1.replace(FILE_PATH, './at/')
        temp = line2.split('/')
        temp_name = temp[2]                                 # get the file name
        if temp_name != NAME:
            NAME = temp_name
            TAG_NUM = TAG_NUM + 1
        replace_str = ';' + str(TAG_NUM) + '\n'             # let the string together
        generate_file.write(line2.replace('\n', replace_str))

    raw_file.close()
    generate_file.close()

def generate_csv_file():
    print ('NEED IMPLEMENTED LATER.')
    # glob.glob("F:/Work_File_share/Face_Recognize_hobo/raw/*/*.jpg")
    # os.system('dir / b / s *.bmp > at.txt')

def main():
    print ('First we wille generate csv file:')
    generate_csv_file()
    print ("Now, add tag in csv file.")
    add_tag_in_csv_file()


if __name__ == '__main__':
    main()