import numpy as np
import cv2
import random

red = (255,0,0)
green = (0,255,0)
blue =(0,0,255)

class ball():
    def __init__(self,env_width,env_height):
        self.radius = 5
        self.step_size = 3
        self.head = random.randint(0,359)
        self.width = env_width
        self.height = env_height
        self.position = random.randint(0,self.width-1),random.randint(0,self.height-1)

    def step(self,turn):
        self.head = self.head+turn
        dx = int(self.step_size * np.cos(self.head*np.pi/180))
        dy = int(self.step_size * np.sin(self.head*np.pi/180))
        new_positon = tuple(np.add(self.position, (dx,dy)))
        if (new_positon[1] > self.height-self.radius or  new_positon[1] < self.radius):
            self.head = - self.head
        elif (new_positon[0] > self.width-self.radius or  new_positon[0] < self.radius):
            self.head = 180 - self.head
        else:
            self.position = new_positon
        return self.position


class env():
    def __init__(self):
        self.num_enemies = 100
        self.width = 800
        self.height = 400
        self.screen = np.zeros((self.height,self.width,3),np.uint8)
        self.red_balls = [ball(self.width,self.height) for b in range(self.num_enemies)]
        self.state = 0
        self.reward = 0

    def reset(self):
        return self.state

    def step(self,a):
        self.screen = np.zeros((self.height, self.width, 3), np.uint8)
        for b in self.red_balls:
            pos = b.step(0)
            cv2.circle(self.screen, b.position, b.radius, red, -1)
        return self.state, self.reward

    def render(self):
        cv2.imshow('screen', self.screen[:,:,::-1])
        cv2.waitKey(1)






if __name__ == '__main__':
    # screen = np.zeros((400,800,3),np.uint8)
    # cv2.circle(screen, (200,200), 100, red, -1)
    # cv2.imshow('screen', screen[:,:,::-1])
    # cv2.waitKey(-1)
    e = env()
    for i in range(10000):
       e.step(0)
       e.render()
