import scrapy
from scrapy.http import Request
import re


class LyricsSpider(scrapy.Spider):
    name = 'scrape_lyrics'
    allowed_domain = ['https://www.metrolyrics.com/']
    start_urls = [f'https://www.metrolyrics.com/{self.name}-alpage-1.html']

    def parse(self, response):

        #Extracting the content using xpath selectors
        artist_song = response.xpath("//h1").extract_first()
        artist = re.search('<title>(.*)</title>', artist_song, re.IGNORECASE).group(1)
        lyrics = response.xpath("//[@class='verse']").extract_all()

        #look through all the artists in the artists page
        #this returns a list of urls
        artists = response.xpath('//a[@class="image"]/@href').getall()
        for artist in artists:
            yield Request(artist)

        #find all their song urls
        #songs1 = response.xpath('//div[@class='module']//a/@href').getall()
        songs = response.xpath('//td/a/@href').getall()
        for song in songs:
            yield Request(song)

        #connect to the url of each song
        #scrape the lyrics from this page
        lyrics = response.xpath('//div[@id="lyrics-body-text"]//text()').getall()
        lyrics = ' '.join(lyrics)
        #write some regex cleaning function here
        lyrics = re.sub("<.+?>", '', str(lyrics)) #takes out HTML code
        lyrics = re.sub("\s+", " ", str(lyrics)) #removes new lines


        item = {'lyrics': lyrics}
        yield item
        print(item)
