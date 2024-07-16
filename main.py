import pygame
import os
import math
import sys
import neat
import time

SCREEN_WIDTH = 1011
SCREEN_HEIGHT = 979

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("Assets","mario_circuit1.png"))

pygame.font.init()
fastest_lap = float('inf')

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.orig_image = pygame.image.load(os.path.join("Assets", "mario_kart_2.png"))
        self.image = self.orig_image
        self.rect = self.image.get_rect(center=(624, 800)) 
        self.velocity = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation = 2.5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.start= False
        self.lap_completed=False
        self.start_time=None
        self.cooldown= 3


    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()
        self.data()
    
    def drive(self):
        self.rect.center += self.velocity * 3


    def collision(self):
        length = 30
        
        col_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                     int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]

        col_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                    int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        
        #Boundary checks
        if SCREEN.get_at(col_right) == pygame.Color(135, 81, 48, 255) or SCREEN.get_at(col_left) == pygame.Color(135, 81, 48, 255):
            self.alive = False

        if SCREEN.get_at(col_right) == pygame.Color(0, 162, 232, 255) or SCREEN.get_at(col_left) == pygame.Color(0, 162, 232, 255):
            current_time=time.time()
            if not self.start:
                self.start = True
                self.start_time= current_time
            elif self.start and not self.lap_completed and (current_time-self.start_time)>self.cooldown:
                self.lap_completed = True

            
        #Collision points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), col_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), col_left, 4)


    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation
            self.velocity.rotate_ip(self.rotation)
        if self.direction == -1:
            self.angle += self.rotation
            self.velocity.rotate_ip(-self.rotation)

        self.image = pygame.transform.rotozoom(self.orig_image, self.angle, 0.08)
        self.rect = self.image.get_rect(center=self.rect.center)

  #Radar lines
    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        
        while not SCREEN.get_at((x, y)) == pygame.Color(135, 81, 48, 255) and length < 125:

            length+=1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (239, 20, 20,0), (x, y), 3)
        
        distance = int(math.sqrt((self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2))
        self.radars.append([radar_angle, distance])

    def data(self):
        input = [0,0,0,0,0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input

def remove(i):
    cars.pop(i)
    ind.pop(i)
    Neurnet.pop(i)

#Evaluation function
def eval_fitness(popul,config):
    global cars,ind,Neurnet,fastest_lap
    cars=[]
    ind=[] 
    Neurnet=[]
    
    for individual_id, individual in popul: 
        cars.append(pygame.sprite.GroupSingle(Car()))
        ind.append(individual)
        net= neat.nn.FeedForwardNetwork.create(individual,config) 
        Neurnet.append(net)
        individual.fitness = 0 
    
    start_time= None
    end_time=None

    run= True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        SCREEN.blit(TRACK, (0, 0)) 

        if len(cars)==0:
            break

        for i,car in enumerate(cars):
            ind[i].fitness+=1
            if not car.sprite.alive:
                remove(i)
    
        for i,car in enumerate(cars):
            output= Neurnet[i].activate(car.sprite.data())
            if output[0]>0.7:
                car.sprite.direction=1
            if output[1]<0.7:
                car.sprite.direction=-1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction=0

            if car.sprite.start and car.sprite.start_time is  not None:
                current_time= time.time()-car.sprite.start_time
            else:
                current_time=0
            if car.sprite.lap_completed:
                lap_time = current_time
                if lap_time < fastest_lap:
                    fastest_lap = lap_time
                run = False  
                break

        #Update
        for car in cars:
            car.draw(SCREEN)
            car.update()

        font = pygame.font.Font('Roboto-Medium.ttf', 20)
        text = font.render(f"Time: {current_time:.2f}", True, (40, 40, 0))
        SCREEN.blit(text, (10, 10))

        if fastest_lap == float('inf'):
            ftext = font.render("Fastest Lap: --:--", True, (40, 40, 0))
        else:
            ftext = font.render(f"Fastest Lap: {fastest_lap:.2f}", True, (40, 40, 0))
        SCREEN.blit(ftext, (10, 40))

        pygame.display.update()

    return max(ind, key=lambda x: x.fitness)

#Neural Network
def run(config_file):
    global population
    config= neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    
    population= neat.Population(config)
    
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)
    population.run(eval_fitness,75)

if __name__=='__main__':
    local_directory= os.path.dirname(__file__)
    configuration_path= os.path.join(local_directory,"config.txt")
    run(configuration_path)
