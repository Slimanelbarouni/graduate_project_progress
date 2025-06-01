#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);
  myServo.attach(9); // Connect servo signal to pin 9
  myServo.write(0);  // Default position (closed)
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'O') {         // Open gate
      myServo.write(90);          // Adjust angle as needed
    } else if (command == 'C') {  // Close gate
      myServo.write(0);           // Reset angle
    }
  }
}
