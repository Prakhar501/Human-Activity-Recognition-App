package com.example.harapp;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import java.util.ArrayList;
import java.util.List;

// Interface for callback
interface SensorDataListener {
    void onSensorDataCollected(float[] sensorData);
}

public class SensorDataCollector implements SensorEventListener {

    private static final String TAG = "SensorCollector";

    private Context context;
    private SensorManager sensorManager;
    private SensorDataListener listener;

    // Sensors
    private Sensor accelerometerSensor;
    private Sensor gyroscopeSensor;
    private Sensor linearAccelerationSensor;

    // Data buffers
    private List<float[]> accelerometerData;
    private List<float[]> gyroscopeData;
    private List<float[]> linearAccelData;

    private int windowSize = 128; // 128 samples per window
    private boolean isCollecting = false;

    // Rate limiting
    private long lastProcessTime = 0;
    private static final long MIN_PROCESS_INTERVAL = 2000; // Process every 2 seconds

    private Handler handler = new Handler(Looper.getMainLooper());

    // Updated constructor with listener
    public SensorDataCollector(Context context, SensorDataListener listener) {
        this.context = context;
        this.listener = listener;
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);

        // Initialize data buffers
        accelerometerData = new ArrayList<>();
        gyroscopeData = new ArrayList<>();
        linearAccelData = new ArrayList<>();
    }

    public void initializeSensors() {
        // Get sensors
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscopeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        linearAccelerationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

        if (accelerometerSensor == null) {
            Log.e(TAG, "Accelerometer sensor not available");
        } else {
            Log.d(TAG, "Accelerometer sensor available");
        }

        if (gyroscopeSensor != null) {
            Log.d(TAG, "Gyroscope sensor available");
        } else {
            Log.e(TAG, "Gyroscope sensor not available");
        }

        if (linearAccelerationSensor != null) {
            Log.d(TAG, "Linear acceleration sensor available");
        } else {
            Log.e(TAG, "Linear acceleration sensor not available");
        }

        Log.d(TAG, "Sensors initialized successfully");
    }

    public void startCollecting() {
        if (!isCollecting) {
            // Clear old data
            accelerometerData.clear();
            gyroscopeData.clear();
            linearAccelData.clear();

            // Register listeners with NORMAL delay (slower to save battery)
            if (accelerometerSensor != null) {
                sensorManager.registerListener(this, accelerometerSensor,
                        SensorManager.SENSOR_DELAY_NORMAL); // Changed from SENSOR_DELAY_GAME
            }
            if (gyroscopeSensor != null) {
                sensorManager.registerListener(this, gyroscopeSensor,
                        SensorManager.SENSOR_DELAY_NORMAL); // Changed from SENSOR_DELAY_GAME
            }
            if (linearAccelerationSensor != null) {
                sensorManager.registerListener(this, linearAccelerationSensor,
                        SensorManager.SENSOR_DELAY_NORMAL); // Changed from SENSOR_DELAY_GAME
            }

            isCollecting = true;
            lastProcessTime = System.currentTimeMillis();
            Log.d(TAG, "Started collecting sensor data at NORMAL rate");
        }
    }

    public void stopCollecting() {
        if (isCollecting) {
            sensorManager.unregisterListener(this);
            isCollecting = false;
            Log.d(TAG, "Stopped collecting sensor data");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        float[] values = event.values.clone();

        switch (event.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                accelerometerData.add(values);
                break;
            case Sensor.TYPE_GYROSCOPE:
                gyroscopeData.add(values);
                break;
            case Sensor.TYPE_LINEAR_ACCELERATION:
                linearAccelData.add(values);
                break;
        }

        // Check if we have enough data AND enough time has passed
        if (accelerometerData.size() >= windowSize &&
                gyroscopeData.size() >= windowSize) {

            long currentTime = System.currentTimeMillis();
            if (currentTime - lastProcessTime >= MIN_PROCESS_INTERVAL) {
                processDataWindow();
                lastProcessTime = currentTime;

                // Keep sliding window (remove oldest to keep size = windowSize)
                if (accelerometerData.size() > windowSize) {
                    accelerometerData.remove(0);
                }
                if (gyroscopeData.size() > windowSize) {
                    gyroscopeData.remove(0);
                }
                if (linearAccelData.size() > windowSize) {
                    linearAccelData.remove(0);
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not needed
    }

    private void processDataWindow() {
        // Rate limiting check (extra safety)
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastProcessTime < MIN_PROCESS_INTERVAL) {
            Log.d(TAG, "Rate limiting: skipping process");
            return;
        }

        // Create input array: 9 channels × 128 samples
        float[] inputData = new float[9 * windowSize];

        // For normalization statistics
        float[] accelSum = new float[3];
        float[] accelSqSum = new float[3];
        float[] gyroSum = new float[3];
        float[] gyroSqSum = new float[3];
        int count = Math.min(windowSize, accelerometerData.size());

        // First pass: calculate mean
        for (int i = 0; i < count; i++) {
            float[] accel = accelerometerData.get(i);
            float[] gyro = gyroscopeData.get(i);

            for (int j = 0; j < 3; j++) {
                accelSum[j] += accel[j];
                accelSqSum[j] += accel[j] * accel[j];
                gyroSum[j] += gyro[j];
                gyroSqSum[j] += gyro[j] * gyro[j];
            }
        }

        // Calculate mean and std
        float[] accelMean = new float[3];
        float[] accelStd = new float[3];
        float[] gyroMean = new float[3];
        float[] gyroStd = new float[3];

        for (int j = 0; j < 3; j++) {
            if (count > 0) {
                accelMean[j] = accelSum[j] / count;
                float accelVariance = (accelSqSum[j] / count) - (accelMean[j] * accelMean[j]);
                accelStd[j] = (float) Math.sqrt(Math.max(accelVariance, 0.001f));

                gyroMean[j] = gyroSum[j] / count;
                float gyroVariance = (gyroSqSum[j] / count) - (gyroMean[j] * gyroMean[j]);
                gyroStd[j] = (float) Math.sqrt(Math.max(gyroVariance, 0.001f));
            } else {
                // Default values if no data
                accelMean[j] = 0;
                accelStd[j] = 1.0f;
                gyroMean[j] = 0;
                gyroStd[j] = 1.0f;
            }

            // Avoid division by zero
            if (accelStd[j] < 0.001f) accelStd[j] = 1.0f;
            if (gyroStd[j] < 0.001f) gyroStd[j] = 1.0f;
        }

        // Second pass: normalize and fill array
        for (int i = 0; i < windowSize; i++) {
            // Default values if sensors are missing
            float[] accel = (i < accelerometerData.size()) ? accelerometerData.get(i) : new float[]{0, 0, 0};
            float[] gyro = (i < gyroscopeData.size()) ? gyroscopeData.get(i) : new float[]{0, 0, 0};
            float[] linear = (i < linearAccelData.size()) ? linearAccelData.get(i) : new float[]{0, 0, 0};

            // Normalize accelerometer data
            float normAccelX = (accel[0] - accelMean[0]) / accelStd[0];
            float normAccelY = (accel[1] - accelMean[1]) / accelStd[1];
            float normAccelZ = (accel[2] - accelMean[2]) / accelStd[2];

            // Normalize gyroscope data
            float normGyroX = (gyro[0] - gyroMean[0]) / gyroStd[0];
            float normGyroY = (gyro[1] - gyroMean[1]) / gyroStd[1];
            float normGyroZ = (gyro[2] - gyroMean[2]) / gyroStd[2];

            // Fill the array
            inputData[i * 9] = normAccelX;     // Normalized Accel X
            inputData[i * 9 + 1] = normAccelY; // Normalized Accel Y
            inputData[i * 9 + 2] = normAccelZ; // Normalized Accel Z
            inputData[i * 9 + 3] = normGyroX;  // Normalized Gyro X
            inputData[i * 9 + 4] = normGyroY;  // Normalized Gyro Y
            inputData[i * 9 + 5] = normGyroZ;  // Normalized Gyro Z
            inputData[i * 9 + 6] = linear[0];  // Linear X (already filtered)
            inputData[i * 9 + 7] = linear[1];  // Linear Y
            inputData[i * 9 + 8] = linear[2];  // Linear Z
        }

        // Debug: Log sample values to see if normalization is working
        if (inputData.length >= 6) {
            Log.d(TAG, "Normalized sample values:");
            Log.d(TAG, String.format("Accel: [%.3f, %.3f, %.3f]",
                    inputData[0], inputData[1], inputData[2]));
            Log.d(TAG, String.format("Gyro:  [%.3f, %.3f, %.3f]",
                    inputData[3], inputData[4], inputData[5]));
            Log.d(TAG, String.format("Stats - Accel mean: [%.3f, %.3f, %.3f], std: [%.3f, %.3f, %.3f]",
                    accelMean[0], accelMean[1], accelMean[2],
                    accelStd[0], accelStd[1], accelStd[2]));
        }

        // Send to MainActivity
        if (listener != null) {
            handler.post(() -> listener.onSensorDataCollected(inputData));
        } else {
            Log.e(TAG, "SensorDataListener is null!");
        }
    }

    public void destroy() {
        stopCollecting();
        sensorManager = null;
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
        }
    }
}