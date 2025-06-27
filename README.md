# TIAGO _gesture memory_ scenario

This metapackage implements a Human-Robot Interaction scenario involving TIAGo robot (PAL Robotics) and a user.

The interaction is based on a gesture-based memory game, in which the robot acts as the game host, mimicking a sequence of letters with its arm. The user has to memorize the sequence, and mimick it back to advance in the game.

The `tiago_gesture_memory_scenario` includes the following packages:
* [`gesture_letter_recognition`](./gesture_letter_recognition/)
* [`tiago_gesture_memory`](./tiago_gesture_memory/)