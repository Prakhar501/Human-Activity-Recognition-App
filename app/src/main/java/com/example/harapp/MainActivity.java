package com.example.harapp;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.LiteModuleLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorDataListener {

    private static final String TAG = "HARApp";
    private static final int PERMISSIONS_REQUEST_CODE = 1001;

    private Module harModule = null;

    // UI Elements
    private TextView statusText;
    private TextView activityText;
    private TextView confidenceText;
    private TextView probabilitiesText;
    private TextView accuracyText;
    private Spinner trueActivitySpinner;
    private Button startButton;
    private Button stopButton;

    private CardView statusCard;
    private CardView activityCard;
    private CardView probabilitiesCard;
    private CardView accuracyCard;

    private SensorDataCollector sensorDataCollector;
    private Handler handler = new Handler(Looper.getMainLooper());

    private boolean isDetecting = false;
    private int correctPredictions = 0;
    private int totalPredictions = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // DEBUG: List ALL assets in detail
        debugListAllAssetsInDetail();

        // Initialize UI first
        initializeUI();
        setupActivitySpinner();
        setupButtonListeners();

        // Check and request permissions
        if (checkPermissions()) {
            initializeApp();
        } else {
            requestPermissions();
        }
    }

    private void debugListAllAssetsInDetail() {
        new Thread(() -> {
            try {
                AssetManager assetManager = getAssets();
                String[] allAssets = assetManager.list("");

                Log.d(TAG, "=========================================");
                Log.d(TAG, "DETAILED ASSET SCAN");
                Log.d(TAG, "=========================================");

                if (allAssets == null || allAssets.length == 0) {
                    Log.e(TAG, "❌ NO ASSETS FOUND IN APK!");
                    return;
                }

                Log.d(TAG, "Total assets found: " + allAssets.length);

                for (int i = 0; i < allAssets.length; i++) {
                    String asset = allAssets[i];

                    try {
                        // Try to open as file
                        InputStream is = assetManager.open(asset);
                        long size = is.available();
                        is.close();

                        Log.d(TAG, String.format("%d. %-50s Size: %8d bytes (%.2f MB)",
                                i + 1, asset, size, size / (1024.0 * 1024.0)));

                        // Check if it's a PyTorch model
                        if (asset.endsWith(".ptl") || asset.endsWith(".pt") || asset.endsWith(".pth")) {
                            Log.d(TAG, "    ⭐ THIS IS A PYTORCH MODEL FILE!");
                        }

                        // Log first few bytes to identify file type
                        if (size > 0) {
                            is = assetManager.open(asset);
                            byte[] header = new byte[4];
                            int read = is.read(header);
                            is.close();

                            if (read >= 4) {
                                String hex = bytesToHex(header);
                                Log.d(TAG, "    First 4 bytes: " + hex);

                                // Check for PyTorch file signature
                                if (hex.startsWith("504B0304")) { // ZIP/PKL file
                                    Log.d(TAG, "    ⚠️ This looks like a ZIP/Pickle file (.pkl, .pt legacy)");
                                } else if (hex.contains("504B")) { // Contains PK (zip)
                                    Log.d(TAG, "    ⚠️ Contains ZIP signature");
                                }
                            }
                        }

                    } catch (IOException e) {
                        // Might be a directory
                        try {
                            String[] subItems = assetManager.list(asset);
                            if (subItems != null && subItems.length > 0) {
                                Log.d(TAG, String.format("%d. %-50s [DIRECTORY] Contains: %d items",
                                        i + 1, asset, subItems.length));

                                // List contents of directory
                                for (String subItem : subItems) {
                                    Log.d(TAG, "        - " + subItem);
                                }
                            }
                        } catch (IOException e2) {
                            Log.d(TAG, String.format("%d. %-50s [ERROR: %s]",
                                    i + 1, asset, e.getMessage()));
                        }
                    }
                }

                Log.d(TAG, "=========================================");

            } catch (IOException e) {
                Log.e(TAG, "Error listing assets: " + e.getMessage());
            }
        }).start();
    }

    private String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02X", b));
        }
        return sb.toString();
    }

    private void initializeUI() {
        statusText = findViewById(R.id.statusText);
        activityText = findViewById(R.id.activityText);
        confidenceText = findViewById(R.id.confidenceText);
        probabilitiesText = findViewById(R.id.probabilitiesText);
        accuracyText = findViewById(R.id.accuracyText);
        trueActivitySpinner = findViewById(R.id.trueActivitySpinner);
        startButton = findViewById(R.id.startButton);
        stopButton = findViewById(R.id.stopButton);

        statusCard = findViewById(R.id.statusCard);
        activityCard = findViewById(R.id.activityCard);
        probabilitiesCard = findViewById(R.id.probabilitiesCard);
        accuracyCard = findViewById(R.id.accuracyCard);
    }

    private void setupActivitySpinner() {
        String[] activities = {
                "Select Activity",
                "WALKING",
                "WALKING_UPSTAIRS",
                "WALKING_DOWNSTAIRS",
                "SITTING",
                "STANDING",
                "LAYING"
        };

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                activities
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        trueActivitySpinner.setAdapter(adapter);

        trueActivitySpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                updateAccuracyDisplay();
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {
                updateAccuracyDisplay();
            }
        });
    }

    private void setupButtonListeners() {
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startDetection();
            }
        });

        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopDetection();
            }
        });
    }

    private boolean checkPermissions() {
        List<String> requiredPermissions = new ArrayList<>();
        requiredPermissions.add(Manifest.permission.BODY_SENSORS);
        requiredPermissions.add(Manifest.permission.ACTIVITY_RECOGNITION);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requiredPermissions.add(Manifest.permission.POST_NOTIFICATIONS);
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            requiredPermissions.add(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS);
        }

        for (String permission : requiredPermissions) {
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void requestPermissions() {
        List<String> permissionsToRequest = new ArrayList<>();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS)
                != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.BODY_SENSORS);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACTIVITY_RECOGNITION)
                != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(Manifest.permission.ACTIVITY_RECOGNITION);
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.POST_NOTIFICATIONS);
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.HIGH_SAMPLING_RATE_SENSORS)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsToRequest.add(Manifest.permission.HIGH_SAMPLING_RATE_SENSORS);
            }
        }

        if (!permissionsToRequest.isEmpty()) {
            ActivityCompat.requestPermissions(this,
                    permissionsToRequest.toArray(new String[0]),
                    PERMISSIONS_REQUEST_CODE);
        } else {
            initializeApp();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }

            if (allGranted) {
                initializeApp();
            } else {
                Toast.makeText(this,
                        "Some permissions were denied. App may not work properly.",
                        Toast.LENGTH_LONG).show();
                statusText.setText("⚠️ Some permissions denied");
                statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
                initializeApp();
            }
        }
    }

    private void initializeApp() {
        statusText.setText("Initializing...");
        statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
        startButton.setEnabled(false);
        stopButton.setEnabled(false);

        new Thread(() -> {
            try {
                // Try to load the model
                loadModel();

                // Initialize sensors if model loaded successfully
                if (harModule != null) {
                    try {
                        sensorDataCollector = new SensorDataCollector(MainActivity.this, MainActivity.this);
                        sensorDataCollector.initializeSensors();
                        Log.d(TAG, "✅ Sensor collector initialized");
                    } catch (Exception e) {
                        Log.e(TAG, "⚠️ Sensor initialization failed: " + e.getMessage());
                    }
                }

            } catch (Exception e) {
                Log.e(TAG, "Error in initializeApp: " + e.getMessage());
            }
        }).start();
    }

    private void loadModel() {
        try {
            Log.d(TAG, "=== STARTING MODEL LOAD ===");

            // First, get ALL assets
            String[] allAssets = getAssets().list("");
            if (allAssets == null || allAssets.length == 0) {
                handler.post(() -> {
                    statusText.setText("❌ No assets found in APK");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
                });
                return;
            }

            Log.d(TAG, "Found " + allAssets.length + " assets:");
            for (String asset : allAssets) {
                Log.d(TAG, "  - " + asset);
            }

            // Try to find ANY model file
            String foundModel = null;

            // Check for PyTorch model files
            for (String asset : allAssets) {
                String lowerAsset = asset.toLowerCase();
                if (lowerAsset.endsWith(".ptl") ||
                        lowerAsset.endsWith(".pt") ||
                        lowerAsset.endsWith(".pth") ||
                        lowerAsset.contains("model") ||
                        lowerAsset.contains("mobilenet") ||
                        lowerAsset.contains("har") ||
                        lowerAsset.contains("uci")) {

                    foundModel = asset;
                    Log.d(TAG, "✅ Found potential model: " + asset);
                    break;
                }
            }

            if (foundModel == null) {
                // Try any file that's not tiny
                for (String asset : allAssets) {
                    try {
                        InputStream is = getAssets().open(asset);
                        long size = is.available();
                        is.close();

                        // If file is > 100KB, it might be a model
                        if (size > 100 * 1024) { // 100KB
                            foundModel = asset;
                            Log.d(TAG, "⚠️ Trying large file as model: " + asset + " (" + size + " bytes)");
                            break;
                        }
                    } catch (Exception e) {
                        // Skip
                    }
                }
            }

            if (foundModel == null) {
                handler.post(() -> {
                    statusText.setText("❌ No model file found");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));

                    // Show what we found
                    StringBuilder message = new StringBuilder("Found assets:\n");
                    for (String asset : allAssets) {
                        message.append("• ").append(asset).append("\n");
                    }

                    Toast.makeText(MainActivity.this, message.toString(),
                            Toast.LENGTH_LONG).show();
                });
                return;
            }

            Log.d(TAG, "Using model file: " + foundModel);

            // Try to load the model
            Module loadedModule = loadModelFile(foundModel);

            if (loadedModule != null) {
                harModule = loadedModule;

                // Create a final copy for the lambda
                final String finalModelName = foundModel;

                handler.post(() -> {
                    statusText.setText("✅ Model " + finalModelName + " loaded");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
                    startButton.setEnabled(true);

                    // Test the model with realistic patterns
                    testModelWithRealisticData();
                });

            } else {
                // Create a final copy for the lambda
                final String finalModelName = foundModel;

                handler.post(() -> {
                    statusText.setText("❌ Failed to load " + finalModelName);
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));

                    // Try manual loading
                    Toast.makeText(MainActivity.this,
                            "Failed to load model: " + finalModelName,
                            Toast.LENGTH_LONG).show();
                });
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in loadModel: " + e.getMessage(), e);
            handler.post(() -> {
                statusText.setText("❌ Error: " + e.getMessage());
                statusText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
            });
        }
    }

    private boolean assetExists(String assetName) {
        try {
            InputStream is = getAssets().open(assetName);
            is.close();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private Module loadModelFile(String modelName) {
        try {
            Log.d(TAG, "Loading model: " + modelName);

            // Step 1: Copy from assets to internal storage
            String modelPath = copyAssetToInternalStorage(modelName);
            Log.d(TAG, "Model copied to: " + modelPath);

            // Step 2: Verify the file
            File modelFile = new File(modelPath);
            if (!modelFile.exists()) {
                Log.e(TAG, "❌ Model file not created");
                return null;
            }

            long fileSize = modelFile.length();
            Log.d(TAG, "Model file size: " + fileSize + " bytes");

            if (fileSize == 0) {
                Log.e(TAG, "❌ Model file is 0 bytes - corrupted!");
                return null;
            }

            // Step 3: Load the model
            Log.d(TAG, "Loading with LiteModuleLoader...");
            Module module = LiteModuleLoader.load(modelPath);
            Log.d(TAG, "✅ LiteModuleLoader.load() successful");

            return module;

        } catch (Exception e) {
            Log.e(TAG, "❌ Error in loadModelFile: " + e.getMessage(), e);
            return null;
        }
    }

    private String copyAssetToInternalStorage(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);

        // Delete if exists (to avoid corrupted files)
        if (file.exists()) {
            boolean deleted = file.delete();
            Log.d(TAG, "Deleted existing file: " + deleted);
        }

        Log.d(TAG, "Copying " + assetName + " to: " + file.getAbsolutePath());

        InputStream is = null;
        FileOutputStream fos = null;

        try {
            // Open asset
            is = getAssets().open(assetName);
            long assetSize = is.available();
            Log.d(TAG, "Asset size: " + assetSize + " bytes");

            // Create output
            fos = new FileOutputStream(file);

            // Copy data
            byte[] buffer = new byte[8192];
            int read;
            long total = 0;

            while ((read = is.read(buffer)) != -1) {
                fos.write(buffer, 0, read);
                total += read;
            }

            fos.flush();
            Log.d(TAG, "✅ Copied " + total + " bytes");

        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (IOException e) {
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                }
            }
        }

        // Verify copy
        if (!file.exists()) {
            throw new IOException("File not created");
        }
        if (file.length() == 0) {
            throw new IOException("File is 0 bytes");
        }

        return file.getAbsolutePath();
    }

    private void testModelWithDummyInput() {
        new Thread(() -> {
            try {
                if (harModule == null) {
                    Log.e(TAG, "Model not loaded for test");
                    return;
                }

                Log.d(TAG, "=== TESTING MODEL WITH DUMMY INPUT ===");

                // Create dummy input [1, 9, 128]
                float[] dummyInput = new float[1 * 9 * 128];
                for (int i = 0; i < dummyInput.length; i++) {
                    dummyInput[i] = (float) (Math.random() * 2 - 1); // Random values between -1 and 1
                }

                // Create tensor
                Tensor inputTensor = Tensor.fromBlob(dummyInput, new long[]{1, 9, 128});
                Log.d(TAG, "Input tensor created, shape: [1, 9, 128]");

                // Run inference
                IValue output = harModule.forward(IValue.from(inputTensor));
                Tensor outputTensor = output.toTensor();
                float[] scores = outputTensor.getDataAsFloatArray();

                Log.d(TAG, "✅ Model test successful!");
                Log.d(TAG, "Output shape: " + Arrays.toString(outputTensor.shape()));
                Log.d(TAG, "Output length: " + scores.length);

                // Calculate softmax probabilities
                float[] probabilities = softmax(scores);
                Log.d(TAG, "Probabilities: " + Arrays.toString(probabilities));

                handler.post(() -> {
                    probabilitiesText.setText("Model test passed! Output: " + scores.length + " classes");
                    Toast.makeText(MainActivity.this, "Model loaded and tested successfully!",
                            Toast.LENGTH_SHORT).show();
                });

            } catch (Exception e) {
                Log.e(TAG, "❌ Model test failed: " + e.getMessage(), e);
                handler.post(() -> {
                    statusText.setText("⚠️ Model test failed");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
                });
            }
        }).start();
    }
    private void testModelWithRealisticData() {
        new Thread(() -> {
            try {
                if (harModule == null) {
                    Log.e(TAG, "Model not loaded for test");
                    return;
                }

                Log.d(TAG, "=== TESTING MODEL WITH REALISTIC NORMALIZED DATA ===");

                // Test 4 different realistic patterns (NORMALIZED)
                testPatternWithNormalizedData("WALKING", createNormalizedWalkingPattern());
                Thread.sleep(500);
                testPatternWithNormalizedData("SITTING", createNormalizedSittingPattern());
                Thread.sleep(500);
                testPatternWithNormalizedData("STANDING", createNormalizedStandingPattern());
                Thread.sleep(500);
                testPatternWithNormalizedData("LAYING", createNormalizedLayingPattern());

                Log.d(TAG, "════════════════════════════════════════");
                Log.d(TAG, "✅ Realistic normalized data tests completed");

            } catch (Exception e) {
                Log.e(TAG, "❌ Model test failed: " + e.getMessage(), e);
                handler.post(() -> {
                    statusText.setText("⚠️ Model test failed");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
                });
            }
        }).start();
    }

    private void testPatternWithNormalizedData(String patternName, float[] inputData) {
        try {
            Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 9, 128});
            Log.d(TAG, "Testing pattern: " + patternName);

            // Run inference
            IValue output = harModule.forward(IValue.from(inputTensor));
            Tensor outputTensor = output.toTensor();
            float[] scores = outputTensor.getDataAsFloatArray();

            // Calculate probabilities
            float[] probabilities = softmax(scores);

            // Find predicted class
            int predictedClass = 0;
            float maxProb = probabilities[0];
            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    predictedClass = i;
                }
            }

            Log.d(TAG, "════════════════════════════════════════");
            Log.d(TAG, "Test '" + patternName + "' -> Predicted: " +
                    getActivityName(predictedClass) + " (prob: " + String.format("%.1f", maxProb * 100) + "%)");
            Log.d(TAG, "All probabilities: " + Arrays.toString(probabilities));

            // Show on UI for easy debugging
            final String result = patternName + " -> " + getActivityName(predictedClass) +
                    " (" + String.format("%.1f", maxProb * 100) + "%)";
            handler.post(() -> {
                if (probabilitiesText != null) {
                    probabilitiesText.append("\n" + result);
                }
            });

        } catch (Exception e) {
            Log.e(TAG, "Test pattern error: " + e.getMessage());
        }
    }

    // Create NORMALIZED patterns (mean=0, std=1 for each channel)
    private float[] createNormalizedWalkingPattern() {
        float[] data = new float[1152];
        for (int i = 0; i < 128; i++) {
            float t = i * 0.1f;
            // Normalized walking pattern: periodic motion around 0
            data[i * 9] = (float) Math.sin(t) * 0.8f;      // Accel X: ±0.8
            data[i * 9 + 1] = (float) Math.cos(t) * 0.6f; // Accel Y: ±0.6
            data[i * 9 + 2] = 0.1f;                       // Accel Z: slight variation
            data[i * 9 + 3] = (float) Math.sin(t) * 0.3f; // Gyro X: ±0.3
            data[i * 9 + 4] = (float) Math.cos(t) * 0.2f; // Gyro Y: ±0.2
            data[i * 9 + 5] = 0.05f;                      // Gyro Z: small constant
            // Linear accel near 0 (already gravity-removed)
            data[i * 9 + 6] = 0.0f;
            data[i * 9 + 7] = 0.0f;
            data[i * 9 + 8] = 0.0f;
        }
        return data;
    }

    private float[] createNormalizedSittingPattern() {
        float[] data = new float[1152];
        for (int i = 0; i < 128; i++) {
            // Sitting: very stable, small random noise
            data[i * 9] = (float) (Math.random() * 0.1 - 0.05);     // Accel X: small noise
            data[i * 9 + 1] = (float) (Math.random() * 0.1 - 0.05); // Accel Y: small noise
            data[i * 9 + 2] = 0.0f;                                 // Accel Z: stable
            data[i * 9 + 3] = (float) (Math.random() * 0.05 - 0.025); // Gyro X: tiny noise
            data[i * 9 + 4] = (float) (Math.random() * 0.05 - 0.025); // Gyro Y: tiny noise
            data[i * 9 + 5] = 0.0f;                                 // Gyro Z: zero
            // Linear accel near 0
            data[i * 9 + 6] = 0.0f;
            data[i * 9 + 7] = 0.0f;
            data[i * 9 + 8] = 0.0f;
        }
        return data;
    }

    private float[] createNormalizedStandingPattern() {
        float[] data = new float[1152];
        for (int i = 0; i < 128; i++) {
            // Standing: very stable, almost no motion
            data[i * 9] = 0.02f;    // Accel X: tiny constant
            data[i * 9 + 1] = -0.01f; // Accel Y: tiny constant
            data[i * 9 + 2] = 0.0f;   // Accel Z: zero
            data[i * 9 + 3] = 0.0f;   // Gyro X: zero
            data[i * 9 + 4] = 0.0f;   // Gyro Y: zero
            data[i * 9 + 5] = 0.0f;   // Gyro Z: zero
            // Linear accel near 0
            data[i * 9 + 6] = 0.0f;
            data[i * 9 + 7] = 0.0f;
            data[i * 9 + 8] = 0.0f;
        }
        return data;
    }

    private float[] createNormalizedLayingPattern() {
        float[] data = new float[1152];
        for (int i = 0; i < 128; i++) {
            // Laying: completely flat, no motion
            data[i * 9] = 0.0f;   // Accel X: zero
            data[i * 9 + 1] = 0.0f; // Accel Y: zero
            data[i * 9 + 2] = 0.0f; // Accel Z: zero (gravity would be on different axis)
            data[i * 9 + 3] = 0.0f; // Gyro X: zero
            data[i * 9 + 4] = 0.0f; // Gyro Y: zero
            data[i * 9 + 5] = 0.0f; // Gyro Z: zero
            // Linear accel near 0
            data[i * 9 + 6] = 0.0f;
            data[i * 9 + 7] = 0.0f;
            data[i * 9 + 8] = 0.0f;
        }
        return data;
    }
    private boolean isModelOutputMeaningful(float[] scores) {
        if (scores == null || scores.length < 6) return false;

        // Check if all scores are identical (model broken)
        boolean allSame = true;
        for (int i = 1; i < scores.length; i++) {
            if (Math.abs(scores[i] - scores[0]) > 0.0001f) {
                allSame = false;
                break;
            }
        }
        if (allSame) {
            Log.w(TAG, "⚠️ WARNING: All output scores are identical! Model might be broken.");
            return false;
        }

        // Check if scores are all very close to 0
        float sumAbs = 0;
        for (float score : scores) {
            sumAbs += Math.abs(score);
        }
        if (sumAbs < 0.1f) {
            Log.w(TAG, "⚠️ WARNING: Output scores are all near zero.");
            return false;
        }

        return true;
    }
    private void startDetection() {
        if (harModule == null) {
            Toast.makeText(this, "Model not loaded", Toast.LENGTH_SHORT).show();
            return;
        }

        isDetecting = true;

        if (sensorDataCollector != null) {
            sensorDataCollector.startCollecting();
        } else {
            // Use dummy data for testing
            startDummyDataSimulation();
        }

        startButton.setEnabled(false);
        stopButton.setEnabled(true);

        statusText.setText("🔍 Detecting...");
        statusText.setTextColor(getResources().getColor(android.R.color.holo_blue_dark));

        activityText.setText("---");
        confidenceText.setText("Confidence: ---%");

        Toast.makeText(this, "Detection started", Toast.LENGTH_SHORT).show();
    }

    private void startDummyDataSimulation() {
        new Thread(() -> {
            int counter = 0;
            String[] activities = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                    "SITTING", "STANDING", "LAYING"};

            while (isDetecting) {
                try {
                    counter++;
                    int activityIndex = counter % 6;

                    // Generate realistic dummy data based on activity
                    float[] dummyData = new float[1152]; // 9 * 128

                    // Different patterns for different activities
                    switch (activityIndex) {
                        case 0: // Walking
                            for (int i = 0; i < dummyData.length; i += 9) {
                                // Simulate walking pattern
                                dummyData[i] = (float) Math.sin(counter * 0.1 + i * 0.01) * 1.5f;
                                dummyData[i + 1] = (float) Math.cos(counter * 0.1 + i * 0.01) * 1.0f;
                                dummyData[i + 2] = 9.8f + (float) Math.random() * 0.3f;
                            }
                            break;
                        case 3: // Sitting
                            for (int i = 0; i < dummyData.length; i += 9) {
                                // Stable with small variations
                                dummyData[i] = (float) (Math.random() * 0.2 - 0.1);
                                dummyData[i + 1] = (float) (Math.random() * 0.2 - 0.1);
                                dummyData[i + 2] = 9.8f;
                            }
                            break;
                        default:
                            // Other activities
                            for (int i = 0; i < dummyData.length; i++) {
                                dummyData[i] = (float) (Math.random() * 0.5 - 0.25);
                            }
                    }

                    // Process the dummy data
                    processInference(dummyData, activities[activityIndex]);

                    Thread.sleep(2000); // 2 second intervals

                } catch (InterruptedException e) {
                    break;
                }
            }
        }).start();
    }

    private void processInference(float[] sensorData, String simulatedActivity) {
        if (harModule == null || !isDetecting) return;

        try {
            // Create tensor
            Tensor inputTensor = Tensor.fromBlob(sensorData, new long[]{1, 9, 128});

            // Run inference
            IValue output = harModule.forward(IValue.from(inputTensor));
            Tensor outputTensor = output.toTensor();
            float[] scores = outputTensor.getDataAsFloatArray();

            // Update UI
            handler.post(() -> {
                updateActivityDisplay(scores);
                updateProbabilitiesDisplay(scores);

                // For accuracy testing if spinner is set
                if (trueActivitySpinner.getSelectedItemPosition() > 0) {
                    int trueClass = trueActivitySpinner.getSelectedItemPosition() - 1;

                    // Find predicted class
                    int predictedClass = 0;
                    float maxScore = scores[0];
                    for (int i = 1; i < scores.length; i++) {
                        if (scores[i] > maxScore) {
                            maxScore = scores[i];
                            predictedClass = i;
                        }
                    }

                    totalPredictions++;
                    if (predictedClass == trueClass) {
                        correctPredictions++;
                    }
                    updateAccuracyDisplay();
                }
            });

        } catch (Exception e) {
            Log.e(TAG, "Inference error: " + e.getMessage());
        }
    }

    private void stopDetection() {
        isDetecting = false;

        if (sensorDataCollector != null) {
            sensorDataCollector.stopCollecting();
        }

        startButton.setEnabled(true);
        stopButton.setEnabled(false);

        statusText.setText("⏹️ Detection stopped");
        statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));

        Toast.makeText(this, "Detection stopped", Toast.LENGTH_SHORT).show();
    }

    // In your updateActivityDisplay() method, add a confidence threshold:
    private void updateActivityDisplay(float[] scores) {
        if (scores == null || scores.length < 6) return;

        int predictedClass = 0;
        float maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                predictedClass = i;
            }
        }

        float[] probabilities = softmax(scores);
        float confidence = probabilities[predictedClass];

        // ADD CONFIDENCE THRESHOLD
        if (confidence < 0.6f) { // 60% confidence threshold
            activityText.setText("UNCERTAIN");
            confidenceText.setText(String.format("Low confidence: %.1f%%", confidence * 100));
            confidenceText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
            return;
        }

        String activity = getActivityName(predictedClass);

        activityText.setText(activity);
        confidenceText.setText(String.format("Confidence: %.1f%%", confidence * 100));

        // Color code based on confidence
        if (confidence > 0.8f) {
            confidenceText.setTextColor(getResources().getColor(android.R.color.holo_green_dark));
        } else if (confidence > 0.6f) {
            confidenceText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
        }
    }

    private void updateProbabilitiesDisplay(float[] scores) {
        if (scores == null || scores.length < 6) return;

        float[] probabilities = softmax(scores);
        StringBuilder sb = new StringBuilder();
        String[] activityNames = {"WALKING", "WALKING_UP", "WALKING_DOWN", "SITTING", "STANDING", "LAYING"};

        for (int i = 0; i < 6; i++) {
            sb.append(String.format("%-15s: %6.1f%%\n",
                    activityNames[i],
                    probabilities[i] * 100));
        }

        probabilitiesText.setText(sb.toString());
    }

    private void updateAccuracyDisplay() {
        if (totalPredictions > 0) {
            double accuracy = (double) correctPredictions / totalPredictions * 100;
            accuracyText.setText(String.format("Accuracy: %d/%d (%.1f%%)",
                    correctPredictions, totalPredictions, accuracy));
        } else {
            accuracyText.setText("Accuracy: 0/0 (0.0%)");
        }
    }

    private float[] softmax(float[] input) {
        float[] output = new float[input.length];
        float sum = 0.0f;

        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i]);
            sum += output[i];
        }

        for (int i = 0; i < output.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    private String getActivityName(int classIndex) {
        switch (classIndex) {
            case 0:
                return "WALKING";
            case 1:
                return "WALKING_UPSTAIRS";
            case 2:
                return "WALKING_DOWNSTAIRS";
            case 3:
                return "SITTING";
            case 4:
                return "STANDING";
            case 5:
                return "LAYING";
            default:
                return "UNKNOWN";
        }
    }

    // SENSOR DATA LISTENER INTERFACE METHOD
    @Override
    public void onSensorDataCollected(float[] sensorData) {
        if (harModule == null || !isDetecting) {
            return;
        }

        new Thread(() -> {
            try {
                // Ensure input is correct size (9 * 128 = 1152)
                if (sensorData.length != 1152) {
                    Log.e(TAG, "Input size mismatch: " + sensorData.length + ", expected 1152");
                    return;
                }

                // Log some sample values for debugging
                Log.d(TAG, "Sample sensor values [0-5]: " +
                        sensorData[0] + ", " + sensorData[1] + ", " + sensorData[2] + ", " +
                        sensorData[3] + ", " + sensorData[4] + ", " + sensorData[5]);

                // Create tensor with shape [1, 9, 128]
                Tensor inputTensor = Tensor.fromBlob(
                        sensorData,
                        new long[]{1, 9, 128}
                );

                // Run inference
                IValue output = harModule.forward(IValue.from(inputTensor));
                Tensor outputTensor = output.toTensor();
                float[] scores = outputTensor.getDataAsFloatArray();

                // Debug: Log raw scores
                Log.d(TAG, "Raw scores: " + Arrays.toString(scores));

                // ADD THIS CHECK - Validate model output
                if (!isModelOutputMeaningful(scores)) {
                    Log.e(TAG, "Model output not meaningful, showing UNCERTAIN");
                    handler.post(() -> {
                        activityText.setText("MODEL ERROR");
                        confidenceText.setText("Check model file");
                        confidenceText.setTextColor(getResources().getColor(android.R.color.holo_red_dark));
                        probabilitiesText.setText("Model outputs are suspicious.\nConsider trying a different model file.");
                    });
                    return;
                }

                // Update UI with prediction
                handler.post(() -> {
                    try {
                        updateActivityDisplay(scores);
                        updateProbabilitiesDisplay(scores);

                        // For accuracy testing if spinner is set
                        if (trueActivitySpinner.getSelectedItemPosition() > 0) {
                            int trueClass = trueActivitySpinner.getSelectedItemPosition() - 1;

                            // Find predicted class
                            int predictedClass = 0;
                            float maxScore = scores[0];
                            for (int i = 1; i < scores.length; i++) {
                                if (scores[i] > maxScore) {
                                    maxScore = scores[i];
                                    predictedClass = i;
                                }
                            }

                            totalPredictions++;
                            if (predictedClass == trueClass) {
                                correctPredictions++;
                            }
                            updateAccuracyDisplay();
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "UI update error: " + e.getMessage());
                    }
                });

            } catch (Exception e) {
                Log.e(TAG, "Prediction error: " + e.getMessage());
                e.printStackTrace();

                // Don't crash - just log and continue
                handler.post(() -> {
                    statusText.setText("⚠️ Prediction error");
                    statusText.setTextColor(getResources().getColor(android.R.color.holo_orange_dark));
                });
            }
        }).start();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (isDetecting && sensorDataCollector != null) {
            sensorDataCollector.startCollecting();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (sensorDataCollector != null) {
            sensorDataCollector.stopCollecting();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (sensorDataCollector != null) {
            sensorDataCollector.destroy();
        }
        isDetecting = false;
    }
}