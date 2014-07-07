# -*- coding: utf-8 -*-
## AGENT BASED MODEL OF CULTURE RETENTION ##

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class agent:

    def __init__(self, row_init, column_init, L, c, q_0):
        self.row = row_init
        self.column = column_init
        self.L = L
        self.c = c
        self.q_0 = q_0
        self.info = [0]*L

        for component in xrange(self.L):
            if np.random.uniform() < 1.0 - self.q_0:
                self.info[component] = 1
        self.q = 1.0*np.sum(self.info)/L

    def move(self, direction):
        if direction == -1:
            pass
        if direction == 0:
            self.row = self.row - 1
        if direction == 1:
            self.column = self.column + 1
        if direction == 2:
            self.row = self.row + 1
        if direction == 3:
            self.column = self.column - 1

    def copy(self, info):
        for component in xrange(self.L):
            if np.random.uniform() < self.c:
                self.info[component] = info[component]
        self.q = 1.0*np.sum(self.info)/self.L

class step:

    def __init__(self, N, L, rho, c, q_0, n_t, d_t):
        self.N = N
        self.L = L
        self.rho = rho
        self.c = c
        self.q_0 = q_0
        self.n_t = n_t
        self.d_t = d_t
        self.lattice = np.array([[0]*N]*N)
        self.agents = []
        
    def populate(self):
        for row in xrange(self.N):
            for column in xrange(self.N):
                if np.random.uniform() < self.rho:
                    self.lattice[row][column] = 1
                    self.agents.append(agent(row,column,self.L,self.c,self.q_0))

    def copy_step(self):
        agents = self.agents[:]
        while len(agents) != 0:
            chosen_one = np.random.choice(agents)
            agents.remove(chosen_one)
            neighbors = neighbors_list(self.N, self.agents, chosen_one)
            hamming = []
            if neighbors == []:
                continue
            else:
                for agent in neighbors:
                    hamming.append(hamming_distance(self.L, chosen_one, agent))
            if min(hamming) == self.L:
                continue
            if hamming.count(min(hamming)) == 1:
                chosen_neighbor = neighbors[hamming.index(min(hamming))]
            else:
                min_indices = []
                for index, value in enumerate(hamming):
                    if value == min(hamming):
                        min_indices.append(index)
                possible_neighbors = [neighbors[i] for i in min_indices]
                chosen_neighbor = np.random.choice(possible_neighbors)
            if np.random.uniform() < (1.0 - (1.0*float(min(hamming))/self.L)):
                chosen_one.copy(chosen_neighbor.info)
        
    def move_step(self):
        agents = self.agents[:]
        while len(agents) != 0:
            chosen_one = np.random.choice(agents)
            agents.remove(chosen_one) 
            mean = 0
            hamming = []
            neighbors = neighbors_list(self.N, self.agents, chosen_one)
            for agent in neighbors:
                hamming.append(hamming_distance(self.L, chosen_one, agent))
            if len(hamming) != 0:
                mean = np.mean(hamming)
            else:
                mean = self.L+1
            if len(neighbors) == 4:
                continue
            else:
                if len(neighbors) > 2 and mean < 0.75*self.L: #neighbor tolerance : 0.5, culture tolerance : .50
                    continue
                else:
                    directions = available_direction(self.N, self.lattice, chosen_one)
                    self.lattice[chosen_one.row][chosen_one.column] = 0
                    chosen_one.move(optimal_direction(self.N, self.L, self.lattice, self.agents, chosen_one, directions))
                    self.lattice[chosen_one.row][chosen_one.column] = 1

    def return_phi(self):
        sum_list = np.array([0]*self.L)
        for agent in self.agents:
            for index, component in enumerate(agent.info):
                sum_list[index] += component
        if 0 in sum_list:
            return 0
        else:
            return 1

    def return_q(self):
        q_ave = 0
        for agent in self.agents:
            q_ave += agent.q
        q_ave = 1.0 * q_ave/len(self.agents)
        return q_ave
    
    def int_information_matrix(self):
        information = np.array([[[0]*self.L]*self.N]*self.N)
        for agent in self.agents:
            for index,component in enumerate(agent.info):
                information[agent.row][agent.column][index] = component
        return information

    def str_information_matrix(self):
        information = np.array([[[0]*self.L]*self.N]*self.N, dtype='str')
        for agent in self.agents:
            for index,component in enumerate(agent.info):
                information[agent.row][agent.column][index] = component
        return information

    def lattice_plot(self, step, specific):
        state_lattice = np.array([[np.nan]*self.N]*self.N)
        if step in specific:
            for row in xrange(self.N):
                for column in xrange(self.N):
                    binary = convert_to_binary(self.str_information_matrix()[row][column])
                    for agent in self.agents:
                        if agent.row == row and agent.column == column:
                            state_lattice[row][column] = binary
            plt.imshow(state_lattice,interpolation="nearest",extent=[0,self.N,0,self.N],norm=matplotlib.colors.normalize(vmin=0,vmax=2**self.L))
            #plt.colorbar()
            plt.xticks(np.arange(self.N+1),[])
            plt.yticks(np.arange(self.N+1),[])
            #plt.grid(ls="solid")
            pad_step = str(step).zfill(4)
            plt.savefig("Lattice_plot_N{0}_L{1}_rho{2}_c{3}_q{4}_nt{5}_dt{6}_t{7}.png".format(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,pad_step))
            plt.close()



class model(step):

    def __init__(self,N,L,rho,c,q_0,n_t,d_t,T,S):
        step.__init__(self,N,L,rho,c,q_0,n_t,d_t)
        self.T = T
        self.S = S

    def return_mean_phi(self):
        return [np.mean(x) for x in self.phi]

    def return_t_half(self):
        mean_phi = self.return_mean_phi()
        for mean in mean_phi:
            if mean <= 0.5:
                return mean
                break
        else:
            return self.T
        
    def plot_phi(self):
        p = open("phi_N{0}_L{1}_rho{2}_c{3}_q{4}_nt{5}_dt{6}_T{7}_S{8}.txt".format(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,self.T,self.S),'w')
        mean_phi = self.return_mean_phi()
        std_phi = [np.std(x)/np.sqrt(self.S) for x in self.phi]
        for index in xrange(len(mean_phi)):
            print >>p, index, mean_phi[index], std_phi[index]
        plt.errorbar(range(0,self.T+1),mean_phi, yerr = std_phi,marker='o')
        plt.ylim(-0.05,1.05)
        plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
        plt.xscale('log')
        plt.xlabel(r'$t$')#,size=32)
        plt.ylabel(r'$\Phi(t)$')#,size=32)
        plt.savefig("phi_N{0}_L{1}_rho{2}_c{3}_q{4}_nt{5}_dt{6}_T{7}_S{8}.png".format(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,self.T,self.S))
        plt.close()
        p.close()

    def plot_q(self):
        q = open("q_N{0}_L{1}_rho{2}_c{3}_q{4}_nt{5}_dt{6}_T{7}_S{8}.txt".format(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,self.T,self.S),'w')
        mean_q_ave = [np.mean(x) for x in self.q_ave]
        std_q_ave = [np.std(x)/np.sqrt(self.S) for x in self.q_ave]
        for index in xrange(len(mean_q_ave)):
            print >>q, index, mean_q_ave[index], std_q_ave[index]
        plt.errorbar(range(0,self.T+1),mean_q_ave, yerr = std_q_ave,marker='o')
        plt.xscale('log')
        plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
        plt.ylim(-0.05,1.05)
        plt.xlabel(r'$t$')#,size=32)
        plt.ylabel(r'$\langle q(t) \rangle$')#,size=32)
        plt.savefig("q_N{0}_L{1}_rho{2}_c{3}_q{4}_nt{5}_dt{6}_T{7}_S{8}.png".format(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,self.T,self.S))
        plt.close()
        q.close()

        
        
    def start(self):
        self.phi = np.array([[0]*self.S]*(self.T+1), dtype="float")
        self.q_ave = np.array([[0]*self.S]*(self.T+1), dtype="float")
        #self.t_half = np.array([0]*(self.T+1), dtype="float")

        s = 0
        while s < self.S:
            self.__init__(self.N,self.L,self.rho,self.c,self.q_0,self.n_t,self.d_t,self.T,self.S)
            step.populate(self)
            self.phi[0][s] = step.return_phi(self)
            self.q_ave[0][s] = step.return_q(self)
            #self.t_half[0] = self.return_t_half()
            if s == 0:
                step.lattice_plot(self, 0,[0])
            t = 1
            while t < self.T + 1:
                step.copy_step(self)
                step.move_step(self)
                self.phi[t][s] = step.return_phi(self)
                self.q_ave[t][s] = step.return_q(self)
                #self.t_half[t] = self.return_t_half()
                if s == 0:
                    step.lattice_plot(self, t,[1,100,250,500,1000])#[1,10,50,100,250,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
                if t % 250 == 0:
                    print "sample=",s,",t=",t
                t = t + 1
            s = s + 1
        self.plot_phi()
        self.plot_q()
        """
        x = open("t_N{0}_L{1}_rho{2}_m{3}_c{4}_q{5}_T{6}_S{7}.txt".format(self.N,self.L,self.rho,self.m,self.c,self.q_0,self.T,self.S),'w')
        for index in xrange(len(self.t_half)):
            print >>x, index, self.t_half[index]
        x.close()
        plt.plot(range(self.T+1),self.t_half,'o')
        plt.xlabel(r'$t$')#,size=32)
        plt.ylabel(r'$t_{1/2}$')#,size=32)
        plt.savefig("t_N{0}_L{1}_rho{2}_m{3}_c{4}_q{5}_T{6}_S{7}.png".format(self.N,self.L,self.rho,self.m,self.c,self.q_0,self.T,self.S))
        plt.close()
        """

def neighbors_list(N, agent_list, agent):
    """returns list of neighboring agents"""
    neighbors = []
    for neighbor in agent_list:
        if agent.row != 0:
            if agent.row - 1 == neighbor.row and agent.column == neighbor.column:
                neighbors.append(neighbor)
        if agent.row != N-1:
            if agent.row + 1 == neighbor.row and agent.column == neighbor.column:
                neighbors.append(neighbor)
        if agent.column != 0:
            if agent.column - 1 == neighbor.column and agent.row == neighbor.row:
                neighbors.append(neighbor)
        if agent.column != N-1:
            if agent.column + 1 == neighbor.column and agent.row == neighbor.row:
                neighbors.append(neighbor)
        if len(neighbors) == 4:
            break
    return neighbors

def hamming_distance(L, agent1, agent2):
    distance = 0
    for index in xrange(L):
        if agent1.info[index] != agent2.info[index]:
            distance += 1
    return distance
    
def available_direction(N, array, agent):
    """returns directions of free sites"""
    neighbor_list = []
    if agent.row != 0:
        if array[agent.row-1][agent.column] == 0:
            neighbor_list.append(0)
    if agent.row != N-1:
        if array[agent.row+1][agent.column] == 0:
            neighbor_list.append(2)
    if agent.column != N-1:
        if array[agent.row][agent.column+1] == 0:
            neighbor_list.append(1)
    if agent.column != 0:
        if array[agent.row][agent.column-1] == 0:
            neighbor_list.append(3)
    else:
        return neighbor_list
    return neighbor_list

def occupied_direction(N, array, agent):
    neighbor_list = []
    if agent.row != 0:
        if array[agent.row-1][agent.column] == 1:
            neighbor_list.append(0)
    if agent.row != N-1:
        if array[agent.row+1][agent.column] == 1:
            neighbor_list.append(2)
    if agent.column != N-1:
        if array[agent.row][agent.column+1] == 1:
            neighbor_list.append(1)
    if agent.column != 0:
        if array[agent.row][agent.column-1] == 1:
            neighbor_list.append(3)
    else:
        return neighbor_list
    return neighbor_list

def occupied_direction_lattice(N, array, row, column):
    neighbor_list = []
    if row != 0:
        if array[row-1][column] == 1:
            neighbor_list.append(0)
    if row != N-1:
        if array[row+1][column] == 1:
            neighbor_list.append(2)
    if column != N-1:
        if array[row][column+1] == 1:
            neighbor_list.append(1)
    if column != 0:
        if array[row][column-1] == 1:
            neighbor_list.append(3)
    else:
        return neighbor_list
    return neighbor_list

def optimal_direction(N,L,array,agent_list,agent,directions):
    direction_payoff = dict()
    if 0 in directions:
        hamming = 0
        move_up_neighbor_directions = occupied_direction_lattice(N,array, agent.row-1,agent.column)
        for direction in move_up_neighbor_directions:
            if direction == 0:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row-2 and value.column == agent.column:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue
            if direction == 1:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row-1 and value.column == agent.column+1:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue                        
            if direction == 3:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row-1 and value.column == agent.column-1:
                        hamming += hamming_distance(L,agent,value)
                        break    
                continue
        if len(move_up_neighbor_directions) == 1:
            hamming = 0
        else:
            hamming = 1.0 * hamming/(len(move_up_neighbor_directions)-1)
        direction_payoff[0] = hamming
    if 1 in directions:
        hamming = 0
        move_right_neighbor_directions = occupied_direction_lattice(N, array, agent.row,agent.column+1)
        for direction in move_right_neighbor_directions:
            if direction == 0:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row-1 and value.column == agent.column+1:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue
            if direction == 1:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row and value.column == agent.column+2:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue                        
            if direction == 2:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+1 and value.column == agent.column+1:
                        hamming += hamming_distance(L,agent,value)
                        break    
                continue
        if len(move_right_neighbor_directions) == 1:
            hamming = 0
        else:
            hamming = 1.0 * hamming/(len(move_right_neighbor_directions)-1)
        direction_payoff[1] = hamming
    if 2 in directions:
        hamming = 0
        move_down_neighbor_directions = occupied_direction_lattice(N,array, agent.row+1,agent.column)
        for direction in move_down_neighbor_directions:
            if direction == 1:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+1 and value.column == agent.column+1:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue                        
            if direction == 2:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+2 and value.column == agent.column:
                        hamming += hamming_distance(L,agent,value)
                        break    
                continue     
            if direction == 3:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+1 and value.column == agent.column-1:
                        hamming += hamming_distance(L,agent,value)
                        break    
                continue     
        if len(move_down_neighbor_directions) == 1:
            hamming = 0
        else:
            hamming = 1.0 * hamming/(len(move_down_neighbor_directions)-1)
        direction_payoff[2] = hamming    
    if 3 in directions:
        hamming = 0
        move_left_neighbor_directions = occupied_direction_lattice(N, array, agent.row,agent.column-1)
        for direction in move_left_neighbor_directions:
            if direction == 0:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row-1 and value.column == agent.column-1:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue
            if direction == 2:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+1 and value.column == agent.column-1:
                        hamming += hamming_distance(L,agent,value)
                        break
                continue                        
            if direction == 3:
                for index,value in enumerate(agent_list):
                    if value.row == agent.row+1 and value.column == agent.column-2:
                        hamming += hamming_distance(L,agent,value)
                        break    
                continue     
        if len(move_left_neighbor_directions) == 1:
            hamming = 0
        else:
            hamming = 1.0 * hamming/(len(move_left_neighbor_directions)-1)
        direction_payoff[3] = hamming
    if len(direction_payoff) != 0:
        if direction_payoff.values().count(min(direction_payoff.values())) == 1:
            return direction_payoff.keys()[direction_payoff.values().index(min(direction_payoff.values()))]
        else:
            direction_choices = []
            for index, value in enumerate(direction_payoff.values()):
                if value == min(direction_payoff.values()):
                    direction_choices.append(index)
            direction_choices = [direction_payoff.keys()[direction_choices[i]] for i in xrange(len(direction_choices))]
            return np.random.choice(direction_choices)
    else:
        return -1

def convert_to_binary(information):
    return int(''.join(information), 2)



"""

=================================
SAMPLE PARAMETERS AND EXPERIMENTS
=================================

N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10

# effect of N
for N in [3,5,7,10,15,20]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()

N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10

# effect of L
for L in [2,4,6,8,10]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
    
N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10

#effect of rho
for rho in [0.25,0.5,0.75, 1]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
    
N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10

# effect of n_t
for n_t in [1,2,3,4]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
    
N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10

# effect of d_t
for n_t in [0.25,0.5,0.75]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
    
N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10
#effect of c
for c in [0.25,0.5,0.75, 1]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
N=10
L=8
rho=0.5
c=0.5
q_0=0.5
n_t = 2
d_t = 0.75
T=1000
S=10
#effect of q_0
for q_0 in [0.25,0.5,0.75, 1]:
    a=model(N,L,rho,c,q_0,n_t,d_t,T,S)
    a.start()
"""

"""
=============
NOTES TO SELF
=============
for future coding:
The simulation of the cultural dynamics is stopped
when the number of links for which 0 < ωij < 1, commonly
called active links, vanishes
"""
