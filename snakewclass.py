import pygame
import random
import os
import time
import neat
import pickle
import numpy as np
from PIL import Image

pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 500
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Snake")

gen = 0
best  = 0

clock = pygame.time.Clock()

timefactor = 1
snake_block = 10
snake_speed = 10*timefactor

class Snake:
    def __init__(self, x, y):
        self.sx = x
        self.sy = y
        self.schangex = 0
        self.schangey = 0
        self.fx = randomx(self.sx)
        self.fy = randomx(self.sy)
        self.snake_length = 1
        self.snake_list = [[self.sx,self.sy]]
        self.direction = ""
        self.lastdirection = ""
        self.directioninno = 0

    def move(self):
        self.sx += self.schangex
        self.sy += self.schangey
        self.snake_list.append([self.sx,self.sy])
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

    def change_direction(self, direction):
        self.lastdirection = self.direction
        self.direction = direction
        if direction == "left":
            self.schangex = -10
            self.schangey = 0
            self.directioninno = 0
        elif direction == "right":
            self.schangex = 10
            self.schangey = 0
            self.directioninno = 1
        elif direction == "up":
            self.schangey = -10
            self.schangex = 0
            self.directioninno = 3
        elif direction == "down":
            self.schangey = 10
            self.schangex = 0
            self.directioninno = 4

def randomx(sx):
    x = round(random.randrange(0, WIN_WIDTH - snake_block) / 10.0) * 10.0
    if x == sx or x == sx+1 or x == sx-1:
        return randomx(sx)
    else:
        return int(x)

def randomy(sy):
    y = round(random.randrange(0, WIN_HEIGHT - snake_block) / 10.0) * 10.0
    if y == sy or y == sy+1 or y == sy-1:
        return randomx(sy)
    else:
        return int(y)

def snakeout(snake):
    screen = []
    for x in range(int(WIN_WIDTH//10*WIN_HEIGHT//10)):
        screen.append(0)
    for x in range(len(snake.snake_list)):
        screen[int((snake.snake_list[x][1]//10*WIN_WIDTH//10)+(snake.snake_list[x][0]//10))] = 0.5
    screen[int(((snake.fy*WIN_WIDTH/10)/10)+(snake.fx/10))] = 1

    #imgscreen = screen
    #x = np.zeros((int(WIN_HEIGHT//10), int(WIN_WIDTH//10+1), 3), dtype=np.uint8)
    #
    #for _x in range(len(imgscreen)):
    #    if (_x+1)//(WIN_HEIGHT//10) > 0:
    #        y = (_x+1)//(WIN_HEIGHT//10)
    #        __x = _x+1 - y*(WIN_HEIGHT//10)
    #    else:
    #        y = 0
    #        __x = _x
    #
    #    if imgscreen[_x] == 0:
    #        x[__x][y] = [0,0,0]
    #    if imgscreen[_x] == 0.5:
    #        x[__x][y]= [128,128,128]
    #    if imgscreen[_x] == 1:
    #        x[__x][y] = [255,255,255]
    #
    #img = Image.fromarray(x, "RGB")
    #img.show()
    return screen

def lastfive(printstr, snakes):
    if len(snakes) < 5:
        print(printstr, end =" ")

def draw_window(win, snakes, best, timefactor):
    win.fill((0,0,0))
    for Snake in snakes:
        pygame.draw.rect(WIN, (255, 255, 255),[Snake.fx,Snake.fy,snake_block,snake_block])
        for x in range(len(Snake.snake_list)):
            pygame.draw.rect(WIN, (128, 128, 128),[Snake.snake_list[x][0],Snake.snake_list[x][1],snake_block,snake_block])

    value = STAT_FONT.render("Best Score: " + str(best), True, (255, 255, 255))
    WIN.blit(value, [0, 0])

    value = STAT_FONT.render("timefactor: " + str(timefactor), True, (255, 255, 255))
    WIN.blit(value, [0, 30])

    pygame.display.update()

def eval_genomes(genomes, config):
    
    should_length = 1

    global WIN, gen, best, timefactor
    win = WIN
    gen += 1

    nets = []
    snakes = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        snakes.append(Snake(round(random.randrange(0, WIN_WIDTH - snake_block) / 10.0) * 10.0,round(random.randrange(0, WIN_HEIGHT - snake_block) / 10.0) * 10.0))
        ge.append(genome)

    start_time = pygame.time.get_ticks()
    run = True
    while run:
        time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i and timefactor != 1:
                    timefactor = timefactor//10
                elif event.key == pygame.K_p:
                    timefactor = timefactor*10

        for snake in snakes:
            snake.move()
            if snake.sx >= WIN_WIDTH or snake.sx < 0 or snake.sy >= WIN_HEIGHT or snake.sy < 0:
                ge[snakes.index(snake)].fitness -= 1
                nets.pop(snakes.index(snake))
                ge.pop(snakes.index(snake))
                snakes.pop(snakes.index(snake))
                lastfive("hit wall", snakes)
        
        for snake in snakes:
            ge[snakes.index(snake)].fitness += 0.001

            output = nets[snakes.index(snake)].activate([snake.sx, WIN_WIDTH-snake.sx, snake.sy, WIN_HEIGHT-snake.sy, snake.fx-snake.sx, snake.fy-snake.sy, snake.directioninno])
            #output = nets[snakes.index(snake)].activate(snakeout(snake))
            maxn = -10
            xnono = 0

            for _x in range(4):
                if output[int(_x)] > maxn:
                    maxn = output[int(_x)]
                    xnono = _x
            
            if xnono == 0:
                snake.change_direction("left")
            elif xnono == 1:
                snake.change_direction("right")
            elif xnono == 2:
                snake.change_direction("up")
            elif xnono == 3:
                snake.change_direction("down")
            
        for snake in snakes:
            if (snake.lastdirection == "left" and snake.direction == "right") or (snake.lastdirection == "right" and snake.direction == "left") or (snake.lastdirection == "up" and snake.direction == "down") or (snake.lastdirection == "down" and snake.direction == "up"):
                ge[snakes.index(snake)].fitness -= 1
                nets.pop(snakes.index(snake))
                ge.pop(snakes.index(snake))
                snakes.pop(snakes.index(snake))
                lastfive("same directior", snakes)
        
        for snake in snakes:
            if snake.sx == snake.fx and snake.sy == snake.fy:
                snake.fx = randomx(snake.sx)
                snake.fy = randomx(snake.sy)
                snake.snake_length += 1
                ge[snakes.index(snake)].fitness += 5

            for _x in snake.snake_list[:-1]:
                if _x == [snake.sx,snake.sy]:
                    ge[snakes.index(snake)].fitness -= 1
                    nets.pop(snakes.index(snake))
                    ge.pop(snakes.index(snake))
                    snakes.pop(snakes.index(snake))
                    lastfive("self coli", snakes)
        if time - start_time > 20000//timefactor:
            for snake in snakes:
                if snake.snake_length <= should_length:
                    nets.pop(snakes.index(snake))
                    ge.pop(snakes.index(snake))
                    snakes.pop(snakes.index(snake))
                    lastfive("its length "+str(snake.snake_length) + " should length " +str(should_length), snakes)
            start_time = time
            should_length += 1

        for snake in snakes:
            if snake.snake_length-1 > best:
                best = snake.snake_length-1


        if len(snakes) == 0:
            print("")
            run = False
            break

        draw_window(WIN, snakes, best, timefactor)
        clock.tick(snake_speed)
        
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes)

    print('\nBest genome:\n{!s}'.format(winner))



if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)