# -*- coding: utf-8 -*-
import scrapy
from ..items import ScrapytestItem
#import re

class TencentSpider(scrapy.Spider):
    name = 'tencent'
    allowed_domains = ['hr.tencent.com']
	# 入口的URLs
#    start_urls = ['https://hr.tencent.com/position.php?keywords=python&lid=2156&tid=0&start=#a0']
#    start_urls = []
#    for i in range(0,100,10):
#        start_urls.append("https://hr.tencent.com/position.php?keywords=python&lid=2156&tid=0&start="
#                          +str(i))
    start_urls = ['https://hr.tencent.com/position.php?keywords=python&lid=2156&tid=0&start=#a0']

    def parse(self, response):
        """
		在系統中獲取到response, 並且精確的提取信息
		"""
        for each in response.xpath("//tr[@class='even']|//tr[@class='odd']"):
            item = ScrapytestItem()
            item['positionName'] = each.xpath('./td[1]/a/text()').extract()[0]
            item['positionLink'] = "https://hr.tencent.com/"+each.xpath('./td[1]/a/@href').extract()[0]
            item['positionType'] = each.xpath('./td[2]/text()').extract()[0]
            yield item
        
        # 使用xpath來匹配下一頁
        # 用待爬的URL再次發起請求
        nextUrl = response.xpath('//*[@id="next"]/@href').extract()[0]
        yield scrapy.Request("https://hr.tencent.com/"+nextUrl, callback=self.parse)
        
        # 用正則表達式來匹配下一頁
        #pattern = re.compile('<a href="(position.php\?keywords=python&lid=2156&tid=0&start=[\d]*?#a)" id="next">下一页</a>')
        #nextUrl = re.findall(pattern, response)
        #if len(nextUrl) > 0:
        #    yield scrapy.Request("https://hr.tencent.com/"+nextUrl[0], callback=self.parse)
        
        
        