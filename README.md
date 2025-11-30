# Pacman CTF Agent

An implementation of Capture the Flag agent for Pacman AI competition.

## Setup

### Installation

1. **Set up the Pacman CTF environment** (follow official [instructions](https://github.com/aig-upf/pacman-eutopia/blob/main/documentation/students.pdf))

2. **Copy the agent files** into the contest directory:
   ```bash
   cp my_team.py pacman-agent/pacman-contest/src/contest/
   cp -r supplementary_materials/ pacman-agent/pacman-contest/src/contest/
   ```

## Usage

### Running a Game

```bash
cd pacman-agent/pacman-contest/src/contest
python capture.py -r my_team -b opponent_team_name
```
