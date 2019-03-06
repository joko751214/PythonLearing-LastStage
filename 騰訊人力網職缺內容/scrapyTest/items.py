# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapytestItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 職位名稱
	positionName = scrapy.Field()
	# 職位詳情鏈接
	positionLink = scrapy.Field()
	# 職位的類別
	positionType = scrapy.Field()
