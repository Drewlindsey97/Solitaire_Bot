#!/bin/bash

# Set Tor proxy
export http_proxy="127.0.0.1:8118"
export http_proxy="127.0.0.1:8118"

# Bot_variables
DELAY_MIN=0.2
DELAY_MAX=0.8
TAP_X=500
TAP_Y=800
SWIPE_X1=300
SWIPE_Y1=500
SWIPE_X2=700
SWIPE_Y2=900
NEXT_GAME_X=800
NEXT_GAME Y=1000

# Function to tap
tap() {
	adb shell input swipe $1 $2 $3 $4
}

# Function to swipe 
swipe () {
	adb shell input swipe $1 $2 $3 $4
}

# Function to random delay
random)_delay() {
	sleep $(($RANDOM % 1000))0.001
}

# Main loop
while true; do
	# Random tap
	tap $TAP_X $TAP_Y
	random_delay

#Random swipe
	swipe $SWIPE_X1 $SWIPE_Y1 $SWIPE_X2 $SWIPE_Y2
	random_delay

	done 

