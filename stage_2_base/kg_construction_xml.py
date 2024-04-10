import xml.sax
import pandas as pd

from collections import defaultdict, Counter

"""
ELEMENT dblp
article
inproceedings
proceedings
book
incollection
phdthesis
mastersthesis
www

"""


class DblpHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentElement = ""
        self.CurrentData = ""

        self.CurrentPaper = None
        self.index = 0
        self.dblp_list = []

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag in ['article', 'inproceedings', 'proceedings', 'book',
                   'incollection', 'phdthesis', 'mastersthesis', 'www']:
            # Element
            self.CurrentElement = tag
            self.CurrentPaper = {
                'address': "",
                'author': [],
                'booktitle': "",
                'cdrom': "",
                'chapter': "",
                'cite': [],
                'crossref': "",
                'data': "",
                'dblp': "",
                'editor': [],
                'ee': "",
                'i': "",
                'isbn': "",
                'journal': "",
                'month': "",
                'note': "",
                'number': "",
                'pages': "",
                'publisher': "",
                'publnr': "",
                'rel': "",
                'school': "",
                'series': "",
                'sub': "",
                'sup': "",
                'title': "",
                'tt': "",
                'url': "",
                'volume': "",
                'year': "",
                'key': "",
                'mdate': ""
            }
            key = attributes['key']
            mdate = attributes['mdate']
            self.CurrentPaper['key'] = key
            self.CurrentPaper['mdate'] = mdate
            # index
            self.index += 1
            print('***' + tag + '_' + str(self.index) + '***')

    # 元素结束事件处理
    def endElement(self, tag):
        if tag in ['article', 'inproceedings', 'proceedings', 'book',
                   'incollection', 'phdthesis', 'mastersthesis', 'www']:
            # list trans str
            self.CurrentPaper['author'] = ' | '.join(self.CurrentPaper['author'])
            self.CurrentPaper['cite'] = ' | '.join(self.CurrentPaper['cite'])
            self.CurrentPaper['editor'] = ' | '.join(self.CurrentPaper['editor'])
            # append
            self.dblp_list.append(self.CurrentPaper)
            self.CurrentElement = ""

        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        content = content.strip()

        if content != "":
            if self.CurrentElement != "" and self.CurrentData in self.CurrentPaper.keys():
                # list
                if self.CurrentData in ['author', 'cite', 'editor']:
                    self.CurrentPaper[self.CurrentData].append(content)
                # str
                else:
                    self.CurrentPaper[self.CurrentData] = content

    # 文档开始事件处理
    def startDocument(self):
        print("Document start")

    def endDocument(self):
        print("Document end")


if __name__ == "__main__":
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = DblpHandler()
    parser.setContentHandler(Handler)

    parser.parse("dblp.xml")

    # write 2 csv
    dblp_df = pd.DataFrame(Handler.dblp_list)
    dblp_df.to_csv('dblp.csv', index=False)
