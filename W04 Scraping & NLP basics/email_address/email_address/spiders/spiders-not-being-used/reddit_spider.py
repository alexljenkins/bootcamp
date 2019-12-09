import scrapy
from scrapy.http import Request
import re


class RedditSpider(scrapy.Spider):
    name = 'reddit_spyder'
    allowed_domain = ['https://www.reddit.com/r/ProgrammerHumor/']
    start_urls = [f'https://www.reddit.com/r/ProgrammerHumor/']


    custom_settings = {
        'DEPTH_LIMIT': 10
    }

    def parse(self, response):
        titles = response.xpath('//*[@class="_eYtD2XCVieq6emjKBH3m"]/text()').extract()
        img_urls = response.xpath('//img[@alt="Post image"]/@src').extract()
        up_votes = response.xpath('//*[@class="_1rZYMD_4xY3gRcSS3p8ODO"]/text()').extract()
        datetimes = response.xpath('//*[@data-click-id="timestamp"]/text()').extract()
        for (title, img_url, up_vote, datetime) in zip(titles, img_urls, up_votes, datetimes):
            yield {'Title': title, 'Image': img_url, 'Up Votes': up_vote, 'Date Time': datetime}

        next_page = response.xpath('//link[@rel="next"]/@href').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
