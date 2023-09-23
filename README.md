# Berkley-UoC-Pacman-AI-Project 🕹️

## Overview

This project is part of the CS487 Introduction to AI course at the **University of Crete**. Originally designed at **Berkeley University**, the project focuses on applying various AI techniques to the classic game of Pacman.

## Phases 🌓

### Phase A

🌟 **Score: 100/100** 🌟

- **Search Algorithms**: DFS, BFS, UCS, and A* implemented for Pacman to navigate mazes.
- **Multi-Agent Problems**: Developed strategies for dealing with multiple agents in the game.
- **Reflex Agents**: Created reflex agents capable of making real-time decisions.
- **Probabilistic Inference**: Implemented probabilistic models for decision-making.

#### How to Run Phase A Examples

```
# Run A* algorithm
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# Run BFS algorithm
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

### Phase B

🌟 **Score: 80/100** 🌟

- **Reflex Agent**: Improved the ReflexAgent to play respectably on various layouts.
- **Minimax**: Implemented a MinimaxAgent that works with any number of ghosts and expands the game tree to an arbitrary depth.
- **Alpha-beta Pruning**: This is the only feature that was not implemented correctly. If you're interested, feel free to try to make it work!
- **Expectimax**: Created an ExpectimaxAgent that models probabilistic behavior of agents who may make suboptimal choices.

#### Specific Challenges in Phase B

- **Evaluation Function**: Developed a better evaluation function for Pacman that cleared the `smallClassic` layout more than half the time.
- **Adversarial Search**: Implemented agents that can handle adversarial search scenarios, including ghosts.
- **Game Tree Exploration**: The project required careful game tree exploration, ensuring the correct number of states were explored.

#### How to Run Phase B Examples

```
# Run ReflexAgent on testClassic layout
python pacman.py -p ReflexAgent -l testClassic

# Run MinimaxAgent with depth 2
python pacman.py -p MinimaxAgent -a depth=2

# Run ExpectimaxAgent on minimaxClassic layout with depth 3
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

## 🏆 Achievements 🏆

- Aced Phase A with a **perfect score of 100/100**!
- Scored **80/100** in Phase B, with room for improvement in Alpha-beta Pruning.
