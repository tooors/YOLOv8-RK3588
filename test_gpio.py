import wiringpi as wp
from wiringpi import GPIO

GPIO_PIN = 7

wp.wiringPiSetup()
wp.pinMode(GPIO_PIN, wp.INPUT)

while True:
    if wp.digitalRead(GPIO_PIN) == wp.LOW:
        print("Button pressed")
    else:
        print("Button not pressed")