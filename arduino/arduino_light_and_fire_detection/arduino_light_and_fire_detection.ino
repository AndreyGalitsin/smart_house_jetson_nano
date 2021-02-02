
#include <TroykaLight.h>
#include <TroykaMQ.h>
#include <ArduinoJson.h>

#define PIN_Light  A0
#define PIN_MQ2  A1

TroykaLight sensorLight(PIN_Light);
MQ2 mq2(PIN_MQ2);
 
void setup()
{
  // открываем последовательный порт
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  mq2.heaterPwrHigh();
  //mq2.calibrate();
  //Serial.print("Ro = ");
  //Serial.println(mq2.getRo());
}


float light_detection()
{
  // считывание данных с датчика освещённости
  sensorLight.read();
  // вывод показателей сенсора освещённости в люксах
  float light = sensorLight.getLightLux();
  return light;
  
}
 
void loop()
{
  float light = light_detection();
  if (!mq2.isCalibrated() && mq2.heatingCompleted()) {
    mq2.calibrate();
  }
  if (mq2.isCalibrated() && mq2.heatingCompleted()) {
    //Serial.println(mq2.readSmoke());
    float smoke = mq2.readSmoke();
    Serial.print("{");
    Serial.print("\"Light\": ");
    Serial.print(light);
    Serial.print(", ");
    Serial.print("\"Smoke\": ");
    Serial.print(smoke);
    Serial.println("}");
  }
  //float light = light_detection();
  //Serial.print("{");
  //Serial.print("\"Light\": ");
  //Serial.print(light);
  //Serial.print(", ");
  //Serial.print("\"Smoke\": ");
  //Serial.print(smoke);
  //Serial.println("}");
  //digitalWrite(LED_BUILTIN, HIGH);
  //detection();
  delay(10);
  }
