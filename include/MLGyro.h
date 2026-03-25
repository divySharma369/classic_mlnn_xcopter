#ifndef ML_GYRO_H
#define ML_GYRO_H

#include <vector>
#include <cmath>

class MLGyroHandler {
public:
    MLGyroHandler(float alpha_init = 0.98f) : alpha(alpha_init) {}

    void processRawData(float accelX, float accelY, float accelZ, float gyroX, float gyroY, float gyroZ, float dt) {
        float accelAngleX = atan2(accelY, accelZ) * 180.0f / M_PI;
        float accelAngleY = atan2(-accelX, sqrt(accelY * accelY + accelZ * accelZ)) * 180.0f / M_PI;

        gyroAngleX += gyroX * dt;
        gyroAngleY += gyroY * dt;

        std::vector<float> nn_input = {accelAngleX, accelAngleY, gyroAngleX, gyroAngleY, gyroX, gyroY};
        std::vector<float> nn_output = predictAngles(nn_input);

        roll = alpha * (roll + gyroX * dt) + (1.0f - alpha) * nn_output[0];
        pitch = alpha * (pitch + gyroY * dt) + (1.0f - alpha) * nn_output[1];
        yaw += gyroZ * dt;
    }

    float getRoll() const { return roll; }
    float getPitch() const { return pitch; }
    float getYaw() const { return yaw; }

private:
    float alpha;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    float gyroAngleX = 0.0f;
    float gyroAngleY = 0.0f;

    std::vector<float> predictAngles(const std::vector<float>& features) {
        float pred_roll = features[0] * 0.5f + features[2] * 0.5f;
        float pred_pitch = features[1] * 0.5f + features[3] * 0.5f;
        return {pred_roll, pred_pitch};
    }
};

#endif
