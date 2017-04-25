import numpy as np
import cv2
import random

red = (255,0,0)
green = (0,255,0)
blue =(0,0,255)
white =(255,255,255)

class ball():
    def __init__(self,env_width,env_height):
        self.radius = 5
        self.step_size = 3
        self.head = random.randint(0,359)
        self.width = env_width
        self.height = env_height
        self.position = random.randint(self.radius,self.width-self.radius-1),random.randint(self.radius,self.height-self.radius-1)

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
        self.num_enemies = 10
        self.width = 800
        self.height = 400
        self.screen = np.zeros((self.height,self.width,3),np.uint8)
        self.red_balls = [ball(self.width,self.height) for b in range(self.num_enemies)]
        self.green_ball = ball(self.width,self.height)
        self.blue_ball = ball(self.width, self.height)
        self.state = 0
        self.reward = 0
        self.view_angles = [-20,-10,0,10,20]
        self.view_len = 10

    def reset(self):
        return self.state

    def step(self, action):
        ''' take one step of the envirment, move red and green balls and calculate reward 
            action: blue ball head angle, between -30 and 30 '''

        # draw red and green balls on the screen
        self.screen = np.zeros((self.height, self.width, 3), np.uint8)
        for b in self.red_balls:
            b.step(0)
            cv2.circle(self.screen, b.position, b.radius, red, -1)
        cv2.circle(self.screen, self.green_ball.position, self.green_ball.radius, green, -1)

        # move blue ball
        pos = self.blue_ball.step(action)
        cv2.circle(self.screen, self.blue_ball.position, self.blue_ball.radius, blue, -1)

        # calculate the reward and state
        self.state = []
        gp = self.green_ball.position
        bp = self.blue_ball.position
        r = self.blue_ball.radius
        
        # calculate the distance between the blue ball and the green ball
        dist = sqrt((gp[0]-bp[0])**2 + (gp[1]-bp[1])**2) 
        self.state.append(dist) 
        
        # find obstacles in viewing angles
        for ang in self.view_angles:
            dx = np.cos(ang * np.pi / 180)
            dy = np.sin(ang * np.pi / 180)
            view_dist = 0
            for i in range(self.view_len):
                posx = int(dx*(r+i))
                posy = int(dy*(r+i))
                # find wall or red ball
                if (posx < 0 or posx > self.width-1 or
                	   posy < 0 or posy > self.height-1 or
                	   self.screen[posx,posy,0] = 255):
                	   view_dist = -i
                	   break
                # find green ball
                elif (self.screen[posx,posy,1] = 255):
                	   view_dist = i
                	   break
            
            self.state.append(view_dist)               	   
        
        # calculate reward 
        
        
        return self.state, self.reward

    def render(self):
        ''' draw the screen for user observation '''
        # draw viewing vectors on the screen
        p = self.blue_ball.position
        r = self.blue_ball.radius
        for ang in self.view_angles:
            dx = np.cos(ang * np.pi / 180)
            dy = np.sin(ang * np.pi / 180)
            for i in range(self.view_len):
                posx = max(min(int(dx*(r+i)), self.width-1), 0)
                posy = max(min(int(dy*(r+i)), self.height-1), 0)
                self.screen[posx,posy,:] = white

        cv2.imshow('screen', self.screen[:,:,::-1])
        cv2.waitKey(1)
        #test
        #test2
        






if __name__ == '__main__':
    e = env()
    for i in range(10000):
       head = random.randint(-30,30)
       e.step(head)
       e.render()
