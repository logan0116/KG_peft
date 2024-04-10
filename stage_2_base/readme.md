## dblp

这个数据比我想象的处理起来要复杂一些

## 第一步

dblp元素列表

统计字段

| 编号 | 元素            | 说明     | 数量      | 包含字段                                                                                                                                                                                |
|----|---------------|--------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | article       | 期刊文章   | 3446881 | author, booktitle, cdrom, cite, crossref, dblp, editor, ee, i, isbn, journal, month, note, number, pages, publisher, publnr, school, series, sub, sup, title, tt, url, volume, year |
| 2  | inproceedings | 会议文章   | 3440054 | address, author, booktitle, cdrom, cite, crossref, editor, ee, i, isbn, journal, month, note, number, pages, publisher, school, series, sub, sup, title, tt, url, volume, year      |
| 3  | proceedings   | 会议录    | 57805   | author, booktitle, cdrom, cite, crossref, editor, ee, i, isbn, journal, note, number, pages, publisher, school, series, sub, sup, title, url, volume, year                          |
| 4  | book          | 书籍     | 20448   | author, booktitle, cdrom, cite, crossref, editor, ee, i, isbn, journal, month, note, pages, publisher, school, series, sub, sup, title, url, volume, year                           |
| 5  | incollection  | 书籍中的文章 | 70314   | author, booktitle, cdrom, chapter, cite, crossref, editor, ee, i, isbn, note, number, pages, publisher, school, series, sub, sup, title, url, volume, year                          |
| 6  | phdthesis     | 博士论文   | 129591  | author, booktitle, crossref, editor, ee, i, isbn, journal, month, note, number, pages, publisher, school, series, sub, sup, title, url, volume, year                                |
| 7  | mastersthesis | 硕士论文   | 23      | author, crossref, data, ee, isbn, month, note, number, pages, publisher, rel, school, series, sup, title, volume, year                                                              |
| 8  | www           | 网页     | 3478845 | author, booktitle, cite, crossref, editor, ee, isbn, journal, month, note, number, pages, publisher, title, url, volume, year                                                       |

## 第二步

导出成csv

## 第三步

捞取引用关系