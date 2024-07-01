import pygame
import random
import neat

pygame.init()


WIDTH, HEIGHT = 1000, 300
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game AI")

DINO_SIZE = (40, 50)
DINO_IMGS = (pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur1.png"), DINO_SIZE)), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur2.png"), DINO_SIZE)), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur3.png"), DINO_SIZE)), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur4.png"), DINO_SIZE)), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur5.png"), (int(1.37 * DINO_SIZE[0]), int(0.63 * DINO_SIZE[1])))), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Dinosaur6.png"), (int(1.37 * DINO_SIZE[0]), int(0.63 * DINO_SIZE[1])))))
GROUND_IMG = pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Ground.jpg"), (1200, 15)))
CACTUS_SIZES = ((28, DINO_SIZE[1] * 1.25), (53, DINO_SIZE[1] * 1.25), (70, DINO_SIZE[1] * 1.25))
CACTUS_IMGS = (pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Cactus1.png"), CACTUS_SIZES[0])), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Cactus2.png"), CACTUS_SIZES[1])), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Cactus3.png"), CACTUS_SIZES[2])))
BIRD_IMGS = (pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets/bird wing up.png"), (35, 30))), pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets/bird wing down.png"), (35, 30))))
RESET_BUTTON = pygame.Surface.convert_alpha(pygame.transform.scale(pygame.image.load("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\Reset.png"), (40, 35)))
reset_rect = pygame.rect.Rect(WIDTH / 2 - 20, HEIGHT / 2, 40, 35)

jump_sound = pygame.mixer.Sound("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\jump.wav")
point_sound = pygame.mixer.Sound("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\point.wav")
die_sound = pygame.mixer.Sound("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\die.wav")


class Dino(pygame.sprite.Sprite):
    MAX_HEIGHT = 20
    ANIMATION_TIME = 3
    ORIGIN_Y = HEIGHT - 22 - DINO_SIZE[1]

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.rect = pygame.rect.Rect(150, self.ORIGIN_Y, DINO_SIZE[0], DINO_SIZE[1])
        self.jumping = False
        self.img_count = 0
        self.vel = 0   # current jump velocity
        self.jump_time = 0   # frames passed since last jump
        self.image = DINO_IMGS[0]
        self.hit = False
        self.crouching = False

    def update(self):
        # PART 1: Taking inputs and handling the jumping and crouching mechanics 
        
        keys = pygame.key.get_pressed()
        if not self.jumping and not self.crouching:
            if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
                jump_sound.play()
                self.vel = -12.5
                self.jump_time = 0
                self.jumping = True

            elif keys[pygame.K_DOWN]:
                self.crouching = True
                self.rect.top += 18.5
        
        if self.jumping and keys[pygame.K_DOWN]:   # speeding up the fall when the player presses down while the dino is in the air
            self.jump_time += 0.4


        self.jump_time += 0.35
        if self.jumping:
            d = (self.vel * self.jump_time + 1.5 * self.jump_time ** 2) / 4

            if d >= 16:
                d = 16
            elif d <= -16:
                d = -16
            self.rect.top = self.rect.top + d

        if self.jumping and self.rect.top > self.ORIGIN_Y:
            self.rect.top = self.ORIGIN_Y
            self.jumping = False



        if not keys[pygame.K_DOWN] and self.crouching:
            self.rect.top -= 18.5
            self.crouching = False

        # PART 2: Updating the image

        self.img_count += 1
        if self.hit:
            self.image = DINO_IMGS[3]

        elif self.jumping:
            self.image = DINO_IMGS[0]

        elif self.img_count < self.ANIMATION_TIME:
            if not self.crouching:
                self.image = DINO_IMGS[1]
            else:
                self.image = DINO_IMGS[4]

        elif self.img_count < self.ANIMATION_TIME * 2:
            if not self.crouching:
                self.image = DINO_IMGS[2]
            else:
                self.image = DINO_IMGS[5]

        else:
            self.img_count = 0
    
    def check_collision(self, sprites):
        dino_mask = pygame.mask.from_surface(self.image)
        
        for sprite in sprites:
            sprite_mask = pygame.mask.from_surface(sprite.image)

            offset_x =  sprite.rect.left - self.rect.left
            offset_y =  sprite.rect.top - self.rect.top

            if dino_mask.overlap(sprite_mask, (offset_x, offset_y)):
                die_sound.play()
                self.hit = True
                return


class Cactus(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        rand_img = random.randint(0, 2)
        self.image = CACTUS_IMGS[rand_img]
        self.rect = pygame.rect.Rect(WIDTH, HEIGHT - 80, CACTUS_SIZES[rand_img][0], CACTUS_SIZES[rand_img][0])

    def update(self, speed):
        self.rect.left -= speed


class Bird(pygame.sprite.Sprite):
    def __init__(self, bird_height=None):
        pygame.sprite.Sprite.__init__(self)
        if bird_height is None:
            bird_height = random.choice((240, 205, 140))

        self.rect = pygame.rect.Rect(WIDTH, bird_height, 35, 20)   # The top of the rect is the height of the bird
        self.image = BIRD_IMGS[0]
           
    def update(self, speed, frames):
        self.rect.left -= speed

        if frames % 30 == 0:
            self.image = BIRD_IMGS[abs(BIRD_IMGS.index(self.image) - 1)]


class Ground:
    def __init__(self, x):
        self.x = x
        self.y = HEIGHT - 30
    
    def move(self, speed):
        self.x -= speed
        if self.x <= -WIDTH:
            self.x = WIDTH


def draw(dino, cacti, grounds, frames, birds=None):
    WIN.fill((255, 255, 255))

    for ground in grounds:
        WIN.blit(GROUND_IMG, (ground.x, ground.y))

    dino.draw(WIN)
    cacti.draw(WIN)
    if birds is not None:
        birds.draw(WIN)

    score = frames / 2.5
    if score % 100 == 0 and score != 0:
        point_sound.play()

    font = pygame.font.SysFont("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\ARCADECLASSIC.TTF", 25)
    text = font.render("SCORE  " + str(int(score)), False, (130, 130, 130))

    WIN.blit(text, (WIDTH - 10 - text.get_width(), text.get_height()))
    
    pygame.display.update()


def reset():
    start_over = False
    while not start_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                start_over = True
            if event.type == pygame.MOUSEBUTTONUP:    
                pos = pygame.mouse.get_pos()
                if reset_rect.collidepoint(pos):
                    main()
        
        font = pygame.font.Font("C:\mmmahrous\Zeiad\Programming\Dinosaur Game AI\Assets\ARCADECLASSIC.TTF", 30)
        WIN.blit(font.render("RETRY", False, (100, 100, 100)), (reset_rect.left - 20, reset_rect.top - 40))
        WIN.blit(RESET_BUTTON, (reset_rect.left, reset_rect.top))
        pygame.display.update()

def main():
    clock = pygame.time.Clock()
    speed = 8
    frames = 0
    player = pygame.sprite.GroupSingle(Dino())
    cacti = pygame.sprite.Group(Cactus())
    birds = pygame.sprite.Group()

    cactus_wait = random.randint(50, 120)
    bird_wait = 100000           # A random frames that we will wait for before spwaning the next obstacle

    ground = [Ground(0), Ground(WIDTH)]

    game_start = False
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP) and game_start == False: 
                game_start = True
                player.jumping = True
            
            elif game_start == False:
                draw(player, cacti, ground, frames)

        if not player.sprites()[0].hit and game_start:
            if frames == 1125:
                bird_wait = 0

            sprites = cacti.sprites()
            if frames >= 1125:   # We have to do everything including birds here otherwise a birds is not defined error will be raised

                if bird_wait == 0 and cactus_wait > 20:
                    cactus_too_close = False
                    for cactus in cacti.sprites():
                        if cactus.rect.left >= 850:
                            cactus_too_close = True

                    if not cactus_too_close:
                        birds.add(Bird())

                        bird_wait = random.randint(180, 480)
                    else:
                        bird_wait = 20

                birds.update(speed, frames)

                for b in birds.sprites():
                    if b.rect.left <= -50:
                        b.kill()

                sprites.extend(birds.sprites())
                
                draw(player, cacti, ground, frames, birds)
            else:
                draw(player, cacti, ground, frames)

            if cactus_wait == 0:
                cacti.add(Cactus())
                cactus_wait = random.randint(30, 120)

            player.sprites()[0].check_collision(sprites)
            player.update()

            ground[0].move(speed)
            ground[1].move(speed)

            cacti.update(speed)

            frames += 1
            bird_wait -= 1
            cactus_wait -= 1

            for c in cacti.sprites():
                if c.rect.left <= -50:
                    c.kill()
            
            if frames % 300 == 0 and speed < 30 and frames > 1200:
                speed += 0.25

        elif game_start:
            reset()
            run = False
    
main()
pygame.quit()
