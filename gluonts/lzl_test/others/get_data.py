#!/usr/bin/python
# -*- coding: UTF-8 -*-


import requests
import re
from tqdm import tqdm



if __name__ == '__main__':

    # 从主页的题库类中，找到所有的题库连接
    url_list = ['http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1436&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1467&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1471&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1484&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1485&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=1486&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=2703&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=4199&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=4200&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=316796&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=409399&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=425321&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=456165&cmd=learning',
                'http://222.200.98.165:8090/redir.php?catalog_id=6&tikubh=458600&cmd=learning'
    ]


    # 不同题库对应的网页
    for url_index, url in enumerate(url_list):

        target_url = url
        html = requests.get(target_url)
        # 定义编码
        html.encoding = 'GBK'

        # 确定截止页数
        url_end_page = re.findall(r"当前第(.*?)页", html.text)
        url_end_page = url_end_page[0].strip().split('/')
        url_end_page = url_end_page[1]


        # 遍历当前种类网页
        with  tqdm(range(1, int(url_end_page)+1)) as it:
            for page in it:

                target_url = url + '&page=' + str(page)
                html = requests.get(target_url)
                html.encoding = 'GBK'

                # 正则单行匹配问题
                question_list = re.findall(r"<div class=\"shiti\"><h3>(.*?)</h3>", html.text)
                # 正则多行匹配答案
                answer_list = re.findall(r"color:#666666\">(.*?)</span>", html.text, flags=re.DOTALL)

                for i in range(len(question_list)):

                    # 去除空格、换行
                    question = question_list[i].strip()
                    answer = answer_list[i].strip().replace('\r', '').replace('\n', '').replace(' ', '')
                    print(question)
                    print(answer)
                    print('\n\n')
