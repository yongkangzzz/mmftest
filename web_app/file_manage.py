import os
from flask import request
import sqlite3
def print_job(object):
    print(object)


def mkdir(path):
    # 引入模块

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + "目录创建成功")
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + "目录已存在")
        return False

def clean(path):
    clear_dir(path)
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        os.rmdir(path_file)

    print("clean finished")

def clear_dir(path):
    isExists = os.path.exists(path)
    if not isExists:
        print("no such dir")
        return
    else:
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                for f in os.listdir(path_file):
                    path_file2 = os.path.join(path_file, f)
                    if os.path.isfile(path_file2):
                        os.remove(path_file2)

def check_database(object):
    db = 'user_info.sqlite3'
    #db = 'sqlite:///user_info.sqlite3'
    con = sqlite3.connect(db)
    cur = con.cursor()
    select_sql = "select file_name from user_info"
    cur.execute(select_sql)
    date_set = cur.fetchall()
    for dir in os.listdir("./static/user"):
        flag = 0
        for row in date_set:
            if dir == row[0]:
                flag = 1
        if flag == 0:
            file_path = './static/user/' + dir
            clear_dir(file_path)
            os.rmdir(file_path)

    cur.close()
    con.close()