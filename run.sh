
#! /bin/bash

export PYTHONPATH=$PYTHONPATH:.

if [ $1 == "d4rl" ]; then
  python3 d4rl/d4rl_main.py --exploration $2 --environment $3 --algorithm $4
elif [ $1 == "fetch" ]; then
  python3 fetch/fetch_main.py --exploration $2 --environment $3 --algorithm $4
elif [ $1 == "hyperMaze" ]; then
  python3 hyperMaze/hyperMaze_main.py --epsilon $2 --algorithm $3
elif [ $1 == "miniGrid" ]; then
  python3 miniGrid/miniGrid_main.py --epsilon $2 --environment $3 --algorithm $4 --use_images $5
else
  echo "wrong argument"
fi