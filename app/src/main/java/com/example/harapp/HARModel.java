package com.example.harapp;

import android.content.Context;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class HARModel {
    private Module model;
    private static final int SEQUENCE_LENGTH = 128;
    private static final int NUM_FEATURES = 9;
    private static final int NUM_CLASSES = 6;

    // Activity labels matching UCI HAR dataset
    private static final String[] ACTIVITY_LABELS = {
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
    };

    public HARModel(Context context) throws IOException {
        model = LiteModuleLoader.load(assetFilePath(context, "har_model.ptl"));
    }

    /**
     * Predict activity from sensor data
     * @param sensorData float array of shape [128, 9]
     * @return PredictionResult containing activity and confidence
     */
    public PredictionResult predict(float[][] sensorData) {
        // Validate input
        if (sensorData.length != SEQUENCE_LENGTH || sensorData[0].length != NUM_FEATURES) {
            throw new IllegalArgumentException(
                    "Expected input shape [128, 9], got [" +
                            sensorData.length + ", " + sensorData[0].length + "]"
            );
        }

        // Flatten to 1D array
        float[] inputArray = new float[SEQUENCE_LENGTH * NUM_FEATURES];
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            System.arraycopy(sensorData[i], 0, inputArray, i * NUM_FEATURES, NUM_FEATURES);
        }

        // Create tensor
        Tensor inputTensor = Tensor.fromBlob(
                inputArray,
                new long[]{1, SEQUENCE_LENGTH, NUM_FEATURES}
        );

        // Run inference
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

        // Apply softmax to get probabilities
        float[] probabilities = softmax(scores);

        // Find max probability
        int maxIndex = 0;
        float maxProb = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }

        return new PredictionResult(
                ACTIVITY_LABELS[maxIndex],
                maxIndex,
                maxProb,
                probabilities
        );
    }

    /**
     * Compute softmax
     */
    private float[] softmax(float[] scores) {
        float[] probabilities = new float[scores.length];
        float maxScore = scores[0];

        // Find max for numerical stability
        for (float score : scores) {
            if (score > maxScore) maxScore = score;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = (float) Math.exp(scores[i] - maxScore);
            sum += probabilities[i];
        }

        // Normalize
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    /**
     * Copy asset to cache directory
     */
    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);

        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public static class PredictionResult {
        public final String activityName;
        public final int activityIndex;
        public final float confidence;
        public final float[] allProbabilities;

        public PredictionResult(String activityName, int activityIndex,
                                float confidence, float[] allProbabilities) {
            this.activityName = activityName;
            this.activityIndex = activityIndex;
            this.confidence = confidence;
            this.allProbabilities = allProbabilities;
        }
    }

    public String[] getActivityLabels() {
        return ACTIVITY_LABELS;
    }
}