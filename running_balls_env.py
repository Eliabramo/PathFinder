import numpy as np
import cv2
import random

red = (255,0,0)
green = (0,255,0)
blue =(0,0,255)
white =(255,255,255)

class ball():
    def __init__(self,env_width,env_height, step_size=3):
        ''' init ball size '''
        self.radius = 10
        self.step_size = step_size
        self.head = random.randint(0,359)
        self.width = env_width
        self.height = env_height
        self.position = random.randint(self.radius,self.width-self.radius-1),random.randint(self.radius,self.height-self.radius-1)

    def step(self,turn):
        ''' turn head by trun angle, and step step_size pixels.
            if hit wall, change head according to Snell law '''
        self.head = self.head+turn
        dx = int(self.step_size * np.cos(self.head*np.pi/180))
        dy = int(self.step_size * np.sin(self.head*np.pi/180))
        new_positon = tuple(np.add(self.position, (dx,dy)))
        if (new_positon[1] > self.height-self.radius-1 or new_positon[1] < self.radius+1):
            self.head = -self.head
        elif (new_positon[0] > self.width-self.radius-1 or new_positon[0] < self.radius+1):
            self.head = 180 - self.head
        else:
            self.position = new_positon
        return self.position


class running_balls_env():
    def __init__(self, num_enemies, enemies_speed):
        self.num_enemies = num_enemies
        self.enemies_step_size = enemies_speed
        self.width = 800
        self.height = 400
        self.max_dist = np.sqrt(self.width**2 + self.height**2)
        self.view_angles = [-40,-20,0,20,40]
        self.view_len = 50
        self.action_size = 3
        self.state_size = len(self.view_angles)+2
        self.step_angle = 5
        self.reset()

    def reset(self):
        self.screen = np.zeros((self.height, self.width, 3), np.uint8)
        self.red_balls = [ball(self.width, self.height, self.enemies_step_size) for b in range(self.num_enemies)]
        self.green_ball = ball(self.width, self.height, 0)
        self.blue_ball = ball(self.width, self.height)
        self.state = np.zeros([1,self.state_size])
        self.d = False
        self.reward = 0
        self.last_dist_step = -1
        self.total_reward = 0
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
        if action == 0:
            turn = -self.step_angle
        elif action == 1:
            turn = 0
        else:
            turn = self.step_angle
        self.blue_ball.step(turn)

        # calculate the state
        self.state = np.zeros([1,self.state_size])
        gp = self.green_ball.position
        bp = self.blue_ball.position
        r = self.blue_ball.radius
        bh = self.blue_ball.head

        # calculate the distance between the blue ball and the green ball
        dx = bp[0]-gp[0]
        dy = gp[1]-bp[1]
        dist = np.sqrt(dx**2 + dy**2)
        self.state[0,0] = dx / self.max_dist          # normalize dist
        self.state[0,1] = dy / self.max_dist  # normalize dist
        dist_step = int(dist/10)

        # find obstacles in viewing angles
        for ang_i in range(len(self.view_angles)):
            ang = self.view_angles[ang_i]
            dx = np.cos((bh+ang) * np.pi / 180)
            dy = np.sin((bh+ang) * np.pi / 180)
            view_dist = 0
            for i in range(self.view_len):
                posx = int(bp[0] + dx*(r+i))
                posy = int(bp[1] + dy*(r+i))
                # find walls or red ball
                if (posx < 0 or posx > self.width-1 or
                    posy < 0 or posy > self.height-1):
                    view_dist = -i
                    break
                # find red balls
                elif self.screen[posy,posx,0] == 255:
                    view_dist = -i
                    break
                # find green ball
                elif self.screen[posy,posx,1] == 255:
                    view_dist = i
                    break

            self.state[0, ang_i+2] = view_dist / self.view_len      # normalize view_dist
        
        # calculate reward
        blue_ball_rect = np.zeros((2*r+1, 2*r+1, 3), np.int32)
        cv2.circle(blue_ball_rect, (r,r), r, blue, -1)
        screen_blue_ball_pos_roi = self.screen[bp[1]-r:bp[1]+r+1, bp[0]-r:bp[0]+r+1, :].astype(np.int32)
        b_plus_r = blue_ball_rect[:,:,2] + screen_blue_ball_pos_roi[:,:,0]
        b_plus_g = blue_ball_rect[:,:,2] + screen_blue_ball_pos_roi[:,:,1]

        # blue ball hit red ball
        if np.any(b_plus_r > 255):
            self.reward = -10
        # blue ball hit green ball
        elif np.any(b_plus_g > 255):
            self.reward = 5000
            self.d = True
        # getting closer to green ball
        elif dist_step < self.last_dist_step:
            self.reward = 5
        # getting away from the green ball
        elif dist_step > self.last_dist_step:
            self.reward = -5
        else:
            self.reward = 0

        self.total_reward += self.reward

        # draw the blue ball
        cv2.circle(self.screen, bp, r, blue, -1)

        self.last_dist_step = dist_step

        return self.state, self.reward, self.d

    def render(self):
        ''' draw the screen for user observation '''
        # draw viewing vectors on the screen
        p = self.blue_ball.position
        r = self.blue_ball.radius
        h = self.blue_ball.head

        for ang in self.view_angles:
            dx = np.cos((h+ang) * np.pi / 180)
            dy = np.sin((h+ang) * np.pi / 180)
            for i in range(self.view_len):
                posx = int(p[0] + dx*(r+i))
                posy = int(p[1] + dy*(r+i))
                if (posx < 0 or posx > self.width-1 or
                    posy < 0 or posy > self.height - 1):
                    break
                else:
                    self.screen[posy,posx,:] = white

        debug_str = 'dx: ' + str(self.state[0,0]) + 'dy: ' + str(self.state[0,1]) + ' v: '
        for r in self.state[0,2:]:
            debug_str += str(r) + ', '
        cv2.putText(self.screen, debug_str, (10, self.height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white)
        cv2.putText(self.screen, str(self.total_reward) + ' (' + str(self.reward) + ')', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, white)

        cv2.imshow('screen', self.screen[:,:,::-1])
        cv2.waitKey(5)

if __name__ == '__main__':
    env = running_balls_env()
    for i in range(10000):
       action = np.argmax(np.random.rand(env.action_size))
       [s,r,d] = env.step(action)
       if d:
           break
       env.render()
