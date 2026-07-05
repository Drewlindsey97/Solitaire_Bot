#!/bin/bash

# Set Tor proxy
export http_proxy="127.0.0.1:8118"
export https_proxy="127.0.0.1:8118"
export no_proxy="localhost,127.0.0.1"

# Bot variables
DELAY_MIN=0.2
DELAY_MAX=0.8
TAP_X=500
TAP_Y=800
SWIPE_X1=300
SWIPE_Y1=500
SWIPE_X2=700
SWIPE_Y2=900
NEXT_GAME_X=800
NEXT_GAME_Y=1000

# Function to tap
tap() {
	adb shell input tap $1 $2
}

# Function to swipe
swipe() {
	adb shell input swipe $1 $2 $3 $4
}

# Function to delay randomly between DELAY_MIN and DELAY_MAX
random_delay() {
	# Generate a random float between DELAY_MIN (0.2) and DELAY_MAX (0.8)
	local r=$(( RANDOM % 600 + 200 )) # random number between 200 and 800 ms
	local delay=$(echo "scale=3; $r / 1000" | bc -l)
	sleep $delay
}

# Main loop
print_status() {
	echo "[*] Tor Proxy set to 127.0.0.1:8118"
	echo "[*] Running simple clicker bot. Press [Ctrl+C] to stop."
}

print_status

while true; do
	# Random tap
	echo "Tapping at ($TAP_X, $TAP_Y)..."
	tap $TAP_X $TAP_Y
	random_delay

	# Random swipe
	echo "Swiping from ($SWIPE_X1, $SWIPE_Y1) to ($SWIPE_X2, $SWIPE_Y2)..."
	swipe $SWIPE_X1 $SWIPE_Y1 $SWIPE_X2 $SWIPE_Y2
	random_delay
done
