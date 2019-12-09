import scrapy
from scrapy.http import Request
import re


class ForumSpider(scrapy.Spider):
    name = 'forum_spyder'
    allowed_domain = ['https://www.pathofexile.com/forum/view-thread/']
    start_urls = [f'https://www.pathofexile.com/forum/view-forum/bug-reports']


    custom_settings = {
        'DEPTH_LIMIT': 10
    }

    def parse(self, response):
        title = response.xpath('//*[@class="topBar last layoutBoxTitle"]/text()').get()
        posts = response.xpath('//*[@class="content"]/text()').getall()
        users = response.xpath('//*[@class="post_by_account"]/text()').getall()
        datetimes = response.xpath('//*[@class="post_date"]/text()').extract()
        # img_urls = response.xpath('//img[@alt="Post image"]/@src').extract()

        for (post, user, datetime) in zip(posts, users, datetimes):
            yield {'Title': title, 'Post': post, 'User': user, 'Date Time': datetime}

        next_page = response.xpath('//link[@rel="Next"]/@href').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
