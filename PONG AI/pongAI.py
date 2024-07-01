import math
import os
import neat
import pygame
import random
import pickle

pygame.init()

WIN_WIDTH, WIN_HEIGHT = 1000, 600
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("PONG")

WHITE = (255, 255, 255)
clock = pygame.time.Clock()
font = pygame.font.SysFont("yugothic", 20)
gen = 0

class Player:
    VEL = 7
    WIDTH = 15
    HEIGHT = 85

    def __init__(self, num):
        self.num = num
        
        if num == 1:
            self.x = 20
        else:
            self.x = WIN_WIDTH - 40

        self.y = 270
        self.score = 0
    
    def move(self, direction):  # A -ve direction means up, a +ve direction means down, 0 means there is no movement
        if not (self.y + self.HEIGHT > WIN_HEIGHT and direction > 0) and not (self.y < 0 and direction < 0):
            self.y += direction * self.VEL
    
    def draw(self):
        pygame.draw.rect(WIN, WHITE, pygame.rect.Rect(self.x, self.y, self.WIDTH, self.HEIGHT))
        
        text = font.render("SCORE: " + str(self.score), False, (0, 0, 255))
        if self.num == 1:
            WIN.blit(text, (10, 10))
        else:
            WIN.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
        

    def win(self, other_player):
        if self.num == 1:
            self.score = 0
            other_player.score = 0
            
        else:
            self.score = 0
            other_player.score = 0

    def reset(self):
        self.y = 270

    def check_scored(self, ball, other_player, score):

        if self.num == 1 and ball.x + ball.SIDE_L > WIN_WIDTH:
            self.score += 1
            
            self.reset()
            other_player.reset()

            ball.x = 490
            ball.y = 290
            ball.m = 0
            ball.direction = random.choice((-1, 1))
        
        elif self.num == 2 and ball.x < 0:
            self.score += 1

            self.reset()
            other_player.reset()

            ball.x = 490
            ball.y = 290
            ball.m = 0
            ball.direction = random.choice((-1, 1))
        
        if self.score == score:
            return True
        return False

class Ball:
    SIDE_L = 20
    VEL = 10
    def __init__(self):
        self.m = random.random() * 2
        self.direction = random.choice((1, -1))
        self.x = 490
        self.y = 290
        self.bounced = False
        self.times_bounced1 = 0   # Times that it bounced off of player 1
        self.times_bounced2 = 0   # Times that it bounced off of player 2   (NEEDED FOR FITNESS CALCULATIONS)

    def move(self):
        if 100 < self.x < 900:
            self.bounced = False

        x_change = math.sqrt((self.VEL ** 2) / (self.m ** 2 + 1)) * self.direction
        self.x += x_change
        self.y += x_change * self.m
    
    def bounce(self, player1, player2):

        if (self.y <= 0 or self.y + self.SIDE_L >= WIN_HEIGHT):
            self.m *= -1

        if self.collide(player1) and not self.bounced:
            self.times_bounced1 += 1
            self.bounced = True

            self.direction *= -1
            self.m = (random.random() + 0.1) * 2 * random.choice((1, -1))
        
        if self.collide(player2) and not self.bounced:
            self.times_bounced2 += 1
            self.bounced = True

            self.direction *= -1
            self.m = (random.random() + 0.1) * 2 * random.choice((1, -1))
    
    def draw(self):
        pygame.draw.rect(WIN, WHITE, pygame.rect.Rect(self.x, self.y, self.SIDE_L, self.SIDE_L))

    def collide(self, player):
        if pygame.rect.Rect(self.x, self.y, self.SIDE_L, self.SIDE_L).colliderect(pygame.rect.Rect(player.x, player.y, player.WIDTH, player.HEIGHT)):
            return True
        return False


def draw_screen(ball, player1, player2):
    WIN.fill((0, 0, 0))

    pygame.draw.rect(WIN, WHITE, pygame.rect.Rect(495, 0, 10, WIN_HEIGHT))

    ball.draw()
    player1.draw()
    player2.draw()

    pygame.display.update()


def main(config):
    run = True

    with open("C:\mmmahrous\Zeiad\Programming\PONG AI/best.pickle", "rb") as f:
        AI_genome = pickle.load(f)
    AI_net = neat.nn.FeedForwardNetwork.create(AI_genome, config)

    player1 = Player(1)
    player2 = Player(2)
    ball = Ball()

    clock = pygame.time.Clock()
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        ball.move()
        ball.bounce(player1, player2)

        output = AI_net.activate((player2.y, ball.y, abs(player2.x - ball.x)))
        decision = output.index(max(output))

        if decision == 1:  
            player2.move(-1)
        elif decision == 2:
            player2.move(1)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            player1.move(-1)
        if keys[pygame.K_DOWN]:
            player1.move(1)

        draw_screen(ball, player1, player2)

        win1 = player1.check_scored(ball, player2, 5)
        win2 = player2.check_scored(ball, player1, 5)
        if win1 or win2:
            run = False


def calc_fitness(genome1, genome2, ball):
    genome1.fitness += ball.times_bounced1
    genome2.fitness += ball.times_bounced2


def simulate(genome1, genome2, config):
    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

    run = True
    player1 = Player(1)
    player2 = Player(2)
    ball = Ball()
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        ball.move()
        ball.bounce(player1, player2)
        # draw_screen(ball, player1, player2)
        # The outputs are in this order: (stay?, move up?, move down?)
        output1 = net1.activate((player1.y, ball.y, abs(player1.x - ball.x)))
        decision1 = output1.index(max(output1))
        # If decision == 0 we dont need to do anything since we just need to stay still
        if decision1 == 1:   # That means that the second output was largest which means we move up
            player1.move(-1)
        elif decision1 == 2:
            player1.move(1)


        output2 = net2.activate((player2.y, ball.y, abs(player2.x - ball.x)))
        decision2 = output2.index(max(output2))

        if decision2 == 1:
            player2.move(-1)
        elif decision2 == 2:
            player2.move(1)

        win1 = player1.check_scored(ball, player2, 1)
        win2 = player2.check_scored(ball, player1, 1)
        
        if win1 or win2 or ball.times_bounced1 > 30:
            calc_fitness(genome1, genome2, ball)
            run = False


def eval_AI(genomes, config):
    global gen
    gen += 1
    ge = []
    nets = []
    for _, g in genomes:
        g.fitness = 0
        ge.append(g)
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
    
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) + 1:
            break
        genome1.fitness = 0

        for genome_id2, genome2 in genomes[i + 1:]:
            if genome1.fitness is None:
                genome1.fitness = 0
            
            # SIMULATION TO TRAIN THE AI AND GET THE BEST ONE
            simulate(genome1, genome2, config)
    if gen % 10 == 0:
        highest = genomes[0][1]
        for _, g in genomes[1:]:
            if g.fitness > highest.fitness:
                highest = g
        with open("bestgen" + str(gen) + ".pickle", "wb") as f:
            pickle.dump(highest, f)


def run(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))   # To printout stats
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_AI, 60)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    directory = os.path.dirname(__file__)
    config_path = os.path.join(directory, "CONFIG.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,   # These r just the headings in the configuration file
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # run(config)
    main(config)
