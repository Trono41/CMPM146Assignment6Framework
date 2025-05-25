from __future__ import annotations
import math
from copy import deepcopy
import time
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose
import random


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None, action=None):
        self.children = {}
        self.parent = parent
        self.results = []
        self.param = param
        self.action = action
        self.actions = {}
        self.visits = 0
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        available_actions = state.get_actions()
        best_score = 0
        best_action = None
        for i in available_actions:
            child = self.children[i.key()]
            score = child.getAverage()
            if score > best_score:
                best_score = score
                best_action = i
        return best_action
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent = 0):
        for child in self.children:
            for i in range(indent):
                print(" ")
            print(self, self.results)
        pass

    def getAverage(self):
        if not self.results:
            return 0.0
        return sum(self.results) / len(self.results)

    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        if state.ended():
            result = self.score(state)
            self.backpropagate(result)
            return

        available = state.get_actions()
        unexplored = [a for a in available if a.key() not in self.children]
        if unexplored:
            self.expand(state, unexplored)
            return
        
        bestChild = None
        bestUCB = -999999999999999
        bestAction = None

        for i in available:
            child = self.children[i.key()]
            UCB = child.getAverage() + (self.param * math.sqrt((math.log(self.visits) if self.visits > 0 else 0.0) / (child.visits)))
            if UCB > bestUCB:
                bestUCB = UCB
                bestChild = child
                bestAction = i

        if bestChild is None:
            bestAction = random.choice(available)
            state.step(bestAction)
            newNode = TreeNode(self.param, parent=self, action=bestAction)
            self.children[bestAction.key()] = newNode
            self.actions[bestAction.key()] = bestAction
            newNode.rollout(state)
            return

        state.step(bestAction)
        bestChild.select(state)

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action ot the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        action = random.choice(available)
        state.step(action)
        n = TreeNode(self.param, parent=self, action=action)
        self.children[action .key()] = n
        self.actions[action .key()] = action 
        n.rollout(state)

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # calls backpropagate with the result you get 
    def rollout(self, state):
        while not state.ended():
            actions = state.get_actions()
            if not actions:
                break
            state.step(random.choice(actions))
        result = self.score(state)
        self.backpropagate(result)
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.visits += 1
        self.results.append(result)
        if self.parent is not None:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: You can start by just using state.score() as the actual value you are 
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state): 
        return state.score() * state.health()
        
        
# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for i in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        bestAction = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if bestAction is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state)) # fallback option
        return bestAction.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
