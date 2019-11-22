"""
GAME CODE TUTORIAL:
https://www.youtube.com/watch?v=-5GNbL33hz0
or with turtle:
https://www.youtube.com/watch?v=gLtQyTF1yX8
"""

import pygame
import random
import os
os.getcwd()
os.chdir('C:\\Alex\\Python\\Pygame\\')
# initialize pygame and create window

WIDTH = 1200
HEIGHT = 800
FPS = 30
# colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# set up assets folders
game_folder = os.path.dirname(os.path.realpath('__file__'))
graphics = os.path.join(game_folder,"Graphics")
p1_graphics = os.path.join(graphics, "Base pack","Player")

class Player(pygame.sprite.Sprite):
    #player sprite
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
#        self.image = pygame.Surface((20,20))
        self.image = pygame.image.load(os.path.join(p1_graphics, "p1_jump.png")).convert()
        self.image.set_colorkey(BLACK)
        #rectangle around a sprite
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH /2, HEIGHT - 50)
        self.speedx = 0

    def update(self):

        # Move left/right
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_a]:
            self.speedx = -10
        if keystate[pygame.K_d]:
            self.speedx = 10
        self.rect.x += self.speedx

        # Don't fall off the edge
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0


class Mob(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((30,40))
        self.image.fill(RED)
        self.rect = self.image.get_rect()

        self.rect.x = (WIDTH - 0.2 * WIDTH)
        self.rect.y = (-900)
        self.speedy = random.randrange(-8,-1)
        self.speedx = random.randrange(-4,4)

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
#        if self.rect.top > HEIGHT + 10 or self.rect.left < -25 or self.rect.right > WIDTH + 20:
#            self.rect.x = random.randrange(WIDTH - self.rect.width)
#            self.rect.y = random.randrange(-100,-40)
#            self.speedy = random.randrange(1,8)


pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My Game")
clock = pygame.time.Clock()


all_sprites = pygame.sprite.Group()

mobs = pygame.sprite.Group()
player = Player()
all_sprites.add(player)

for i in range(8):
    m = Mob()
    all_sprites.add(m)
    mobs.add(m)


#Game loop
running = True
while running:
    #run at speed
    clock.tick(FPS)

    #process input (events)
    for event in pygame.event.get():
        #check for exit
        if event.type == pygame.QUIT:
            running = False

    #update
    all_sprites.update()

    # draw/render
    screen.fill(BLACK)
    all_sprites.draw(screen)
    # do this last. Flips the display
    pygame.display.flip()

pygame.quit()
