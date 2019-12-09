import scrapy

class SampleItem(scrapy.Item):
    images = scrapy.Field()
    image_urls = scrapy.Field()
