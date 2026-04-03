
import pygame
import random


WIDTH = 360
HEIGHT = 480
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Neural Network Simulator")
clock = pygame.time.Clock()

running = True
while running:

    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #3 Draw/render
    screen.fill(BLACK)

    pygame.display.flip()       

pygame.quit()