from flask import Flask, render_template, request
import numpy as np
import random
import json

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/play", methods=['POST'])
def play():
    print('começou')
    info = json.dumps(request.form)
    info_dict = json.loads(str(info))
    print(info_dict)

    class environment():
        def __init__(self, size, goal_reward, treasure_reward, move_cost, goal_location, treasure_location):
            self.width = size
            self.height = size

            #Goal
            self.goal_x = goal_location[0]
            self.goal_y = goal_location[1]
            self.goal_reward = goal_reward
            self.goal = (self.goal_x, self.goal_y)

            #Treasure
            self.treasure_x = treasure_location[0]
            self.treasure_y = treasure_location[1]
            self.treasure = (self.treasure_x, self.treasure_y)
            self.treasure_reward = treasure_reward

            self.move_cost = move_cost
            self.min_moves = (size - 1) * 2

            #Game Control
            self.game_end = False

            #Agent
            self.agent_x = 1
            self.agent_y = 1
            self.agent_has_treasure = False
            self.agent_score = 0    

        def checking_goal(self):
            if self.agent_x == self.goal_x and self.agent_y == self.goal_y:
                self.agent_score += self.goal_reward
                self.game_end = True

        def checking_treasure(self):
            if self.agent_x == self.treasure_x and self.agent_y == self.treasure_y and self.agent_has_treasure == False:
                self.agent_score += self.treasure_reward
                self.agent_has_treasure = True

        def agent_move_up(self):
            if self.game_end == True:
                return 'END OF THE GAME'
            if self.agent_y < self.height:
                self.agent_y += 1
            self.agent_score -= self.move_cost
            self.checking_treasure()
            self.checking_goal()

        def agent_move_right(self):
            if self.game_end == True:
                return 'END OF THE GAME'
            if self.agent_x < self.width:
                self.agent_x += 1
            self.agent_score -= self.move_cost
            self.checking_treasure()
            self.checking_goal()

        def agent_move_down(self):
            if self.game_end == True:
                return 'END OF THE GAME'
            if self.agent_y > 1:
                self.agent_y -= 1
            self.agent_score -= self.move_cost
            self.checking_treasure()
            self.checking_goal()

        def agent_move_left(self):
            if self.game_end == True:
                return 'END OF THE GAME'
            if self.agent_x > 1:
                self.agent_x -= 1
            self.agent_score -= self.move_cost
            self.checking_treasure()
            self.checking_goal()

        def play_again(self):
            self.agent_x = 1
            self.agent_y = 1
            self.agent_score = 0
            self.agent_has_treasure = False
            self.game_end = False

    def print_environment(agent, environment):
        columns = ''
        for i in range(environment.width, 0, -1):
            for g in range(1, environment.height + 1):
                padding = '     '
                empty = True
                for t in agent.track:
                    if (g, i) == t and empty == True:
                        columns += padding[:-1] + 'X'
                        empty = False
                
                if environment.goal_x == g and environment.goal_y == i:
                    if empty == False:
                        columns += 'G'
                    else:
                        columns += padding + 'G'
                    empty = False

                if environment.treasure_x == g and environment.treasure_y == i:
                    if empty == False:
                        columns += 'T'
                    else:
                        columns += padding + 'T'
                    empty = False
                
                if empty == True:
                    columns += padding + '_'
            columns += '\n'
        print(columns)

    class agent():
        def __init__(self, environment):
            pass

        #Percepts
        def percepts(self, environment):
            self.score = environment.agent_score
            self.position = (environment.agent_x, environment.agent_y)

        #Actions
        def move_up(self, environment):
            environment.agent_move_up()
            self.percepts(environment)

        def move_right(self, environment):
            environment.agent_move_right()
            self.percepts(environment)
    
        def move_down(self, environment):
            environment.agent_move_down()
            self.percepts(environment)

        def move_left(self, environment):
            environment.agent_move_left()
            self.percepts(environment)

        def find_neighbors(self, environment, position):
            neighbors = []
            if position[1] < environment.height:
                label_up = (position[0], position[1]+1)
                neighbors.append(label_up)
            if position[0] < environment.width:
                label_right = (position[0]+1, position[1])
                neighbors.append(label_right)
            if position[1] > 1:
                label_down = (position[0], position[1]-1)
                neighbors.append(label_down)
            if position[0] > 1:
                label_left = (position[0]-1, position[1])
                neighbors.append(label_left)

            return neighbors

        def find_move(self, position, goal):
            x_agent = position[0]
            y_agent = position[1]
            x_goal = goal[0]
            y_goal = goal[1]

            if x_agent == x_goal and y_agent < y_goal:
                return 0

            if x_agent == x_goal and y_agent > y_goal:
                return 2

            if y_agent == y_goal and x_agent > x_goal:
                return 3

            if y_agent == y_goal and x_agent < x_goal:
                return 1

        def neighbors_values(self, environment, position):
            neighbors = self.find_neighbors(environment, position)
            neighbor_values_dict = {}
            for i in neighbors:
                neighbor_values_dict[i] = self.value[i]
            return neighbor_values_dict

        def best_neighbor(self, environment, position):
            neighbors_values_dict = self.neighbors_values(environment, position)
            n_values = []
            for i in neighbors_values_dict:
                n_values.append(neighbors_values_dict[i])

            max_n = max(n_values)

            for i in neighbors_values_dict:
                if neighbors_values_dict[i] == max_n:
                    return i
  
        def build_value(self, environment):
            value = {}
            for w in range(1, environment.width + 1):
                for h in range(1, environment.height + 1):
                    new_tuple = (w, h)
                    value[new_tuple] = 0
            return value

        def update_value(self, environment, gama):
            count = self.score
            for i in range (0, len(self.track)):
                count = count * gama
                last_step = self.track[(len(self.track) - 1) - i]
                self.value[last_step] = (self.value[last_step] + count) / 2 #This is the most important algorithm

        def build_policy(self, environment):
            policy = {}
            for w in range(1, environment.width + 1):
                for h in range(1, environment.height + 1):
                    new_tuple = (w, h)
                    policy[new_tuple] = -1 #Random
            return policy

        def update_policy(self, environment):
            for i in self.policy:
                best_n = self.best_neighbor(environment, i)
                self.policy[i] = self.find_move(i, best_n)

        def follow_policy(self, position):
            direction = self.policy[position]
            return direction

        def random_move(self, environment):
            choice = np.random.randint(0, 4)
            if choice == 0:
                self.move_up(environment)
                self.track.append(self.position)

            if choice == 1:
                self.move_right(environment)
                self.track.append(self.position)

            if choice == 2:
                self.move_down(environment)
                self.track.append(self.position)

            if choice == 3:
                self.move_left(environment)
                self.track.append(self.position)

        def policy_move(self, environment):
            direction = self.follow_policy(self.position)

            if self.loop_detect(environment) == False:

                if direction == 0:
                    self.move_up(environment)
                    self.track.append(self.position)

                if direction == 1:
                    self.move_right(environment)
                    self.track.append(self.position)

                if direction == 2:
                    self.move_down(environment)
                    self.track.append(self.position)

                if direction == 3:
                    self.move_left(environment)
                    self.track.append(self.position)

                if direction == -1:
                    self.random_move(environment)
                
            else:
                self.random_move(environment)

        def loop_detect(self, environment):
            if len(self.track) > 2:
                if self.track[len(self.track) - 3] == self.position:
                    return True
                else:
                    if len(self.track) > (environment.width * environment.height) * 2:
                        return True
                    else:
                        return False

        ### NAIVE AGENT ###
        def naive_agent(self, environment, gama):
            self.percepts(environment)
            self.policy = self.build_policy(environment)
            self.value = self.build_value(environment)
            self.track = []

        def play_again(self, environment, epsilon):
            play_list = ['random', 'policy']
            play = np.random.choice(play_list, 1, p=[epsilon, (1 - epsilon)])
            return play

        def play_random(self, environment):
            environment.play_again()
            self.percepts(environment)
            self.track = [self.position]

            count_moves = 0

            while environment.game_end == False:
                self.random_move(environment)
                count_moves += 1

            self.update_value(environment, gama)
            self.update_policy(environment)
            return count_moves

        def play_policy(self, environment):
            environment.play_again()
            self.percepts(environment)
            self.track = [self.position]

            count_moves = 0

            while environment.game_end == False:
                self.policy_move(environment)
                count_moves += 1

            self.update_value(environment, gama)
            self.update_policy(environment)
            return count_moves

    episodes = int(info_dict['episodes'])
    epsilon = float(info_dict['epsilon'])
    gama = float(info_dict['gama'])
    size = int(info_dict['size'])
    move_cost = int(info_dict['move_cost'])
    goal_location = (int(info_dict['goal_location_x']), int(info_dict['goal_location_y']))
    treasure_location = (int(info_dict['treasure_location_x']), int(info_dict['treasure_location_y']))
    goal_reward = int(info_dict['goal_reward'])
    treasure_reward = int(info_dict['treasure_reward']) #No rewards for passing through the treasure

    new_environment = environment(size, goal_reward, treasure_reward, move_cost, goal_location, treasure_location)
    new_agent = agent(new_environment)
    new_agent.naive_agent(new_environment, gama)
    steps = []
    tracks = []
    count_treasure = 0

    for i in range(0, episodes):
        if i == 0:
            run_episode = new_agent.play_random(new_environment)
            label = 'new_agent.play_random(new_environment)'
        else:
            result_function = str(list(new_agent.play_again(new_environment, epsilon))[0])
            label = 'new_agent.play_' + result_function + '(new_environment)'
            run_episode = eval(label)
        steps.append(run_episode)
        print('-----------------')
        print('Episode: ' + str(i))
        print('Label: ' + str(label)[15:21])
        print('Steps: ' + str(steps[i]))
        print('Score: ' + str(new_agent.score))
        if steps[i] == new_environment.min_moves:
            print('Shortest Path: TRUE')
        else:
            print('Shortest Path: FALSE')
        if new_environment.treasure in new_agent.track:
            print('Treasure: TRUE')
            count_treasure += 1
        else:
            print('Treasure: FALSE')
        print_environment(new_agent, new_environment)
        tracks.append(new_agent.track)

    min_steps = new_environment.min_moves
    count_min_steps = 0
    for i in steps:
        if i == min_steps:
            count_min_steps += 1
    print(steps)
    print('Shortest path lenght: ' + str(min_steps))
    print('Percentual shortest path: ' + str(((count_min_steps/episodes)*100)) + '%')
    print('Percentual treasure: ' + str(((count_treasure/episodes)*100)) + '%')

    response = {}
    response['shortest_path_percent'] = str(((count_min_steps/episodes)*100))
    response['treasure_percent'] = str(((count_treasure/episodes)*100))
    response['tracks'] = tracks
    response['size'] = size
    response['goal_location'] = goal_location
    response['treasure_location'] = treasure_location
    json_response = json.dumps(response)
    #print(json_response)
    return render_template('play.html', response=json_response)
  

if __name__ == '__main__':
    app.run()