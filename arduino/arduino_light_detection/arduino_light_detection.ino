
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

void detection()
{
  float light = light_detection();
  float lgp;
  float methane;
  float smoke;
  float hydrogen;
  smoke = fire_detection_v2();

  //StaticJsonDocument<200> jsonDocument;
  //jsonDocument["Light"] = light;
  //jsonDocument["LGP"] = lgp;
  //jsonDocument["Methane"] = methane;
  //jsonDocument["Smoke"] = smoke;
  //jsonDocument["Hydrogen"] = hydrogen;
  //char buffer[200];
  //serializeJsonPretty(jsonDocument, buffer);

  Serial.print("{");
  Serial.print("\"Light\": ");
  Serial.print(light);
  Serial.print(", ");
  
  //Serial.print("\"LGP\": ");
  //Serial.print(lgp);
  //Serial.print(", ");
  
  //Serial.print("\"Methane\": ");
  //Serial.print(methane);
  //Serial.print(", ");
  
  Serial.print("\"Smoke\": ");
  Serial.print(smoke);
  //Serial.print(", ");
  
  //Serial.print("\"Hydrogen\": ");
  //Serial.print(hydrogen);
  
  Serial.println("}");
  
}
float light_detection()
{
  // считывание данных с датчика освещённости
  sensorLight.read();
  // вывод показателей сенсора освещённости в люксах
  float light = sensorLight.getLightLux();
  return light;
  
}

float fire_detection()
{
  // выводим отношения текущего сопротивление датчика
  // к сопротивлению датчика в чистом воздухе (Rs/Ro)
  //Serial.print("Ratio: ");
  //Serial.print(mq2.readRatio());
  // выводим значения газов в ppm
  float lgp = mq2.readLPG();
  float methane = mq2.readMethane();
  float smoke = mq2.readSmoke();
  float hydrogen = mq2.readHydrogen();
  return lgp, methane, smoke, hydrogen;

}

float fire_detection_v2()
{
  // выводим отношения текущего сопротивление датчика
  // к сопротивлению датчика в чистом воздухе (Rs/Ro)
  //Serial.print("Ratio: ");
  //Serial.print(mq2.readRatio());
    // если прошёл интервал нагрева датчика
  // и калибровка не была совершена
  if (!mq2.isCalibrated() && mq2.heatingCompleted()) {
    // выполняем калибровку датчика на чистом воздухе
    mq2.calibrate();
    // выводим сопротивление датчика в чистом воздухе (Ro) в serial-порт
    //Serial.print("Ro = ");
    //Serial.println(mq2.getRo());
  }
  if (mq2.isCalibrated() && mq2.heatingCompleted()) {
    // выводим значения газов в ppm
    float lgp = mq2.readLPG();
    float methane = mq2.readMethane();
    float smoke = mq2.readSmoke();
    float hydrogen = mq2.readHydrogen();
    //return lgp, methane, smoke, hydrogen;
    return smoke;
  }
}
 
void loop()
{
  digitalWrite(LED_BUILTIN, HIGH);
  detection();
  delay(1000);
  }
