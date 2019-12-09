import scrapy
from scrapy.http import Request

# have to create a class for the spider
#variable names set and required
class EmailSpider(scrapy.Spider):
    name = 'scrape_emails'  # name used to run the spider
    allowed_domains = ['coingecko.com']  # domain it's allowed to crawl
    start_urls = ['https://www.coingecko.com/en/']  # starting point

    custom_settings = {
        'DEPTH_LIMIT': 10,  # not required, but layers
    }

    #main function for the spider
    def parse(self, response):

        prices = response.xpath('//td[@class="td-price price text-right"]//a/span[@class="no-wrap"]/text()').getall()
        coins = response.xpath('//td[@class="py-0 coin-name"]/@data-sort').getall()

        if len(coins) == len(prices):
            for c, p in zip(coins, prices):
                item = {'coin': c, 'price': p}
                print(item)
                yield item

        next_url = response.xpath('//a[@rel="next"]/@href').get()
        next_url = response.urljoin(next_url)
        print(next_url)
        yield Request(next_url)
