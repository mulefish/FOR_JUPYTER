const fs = require('fs');
const mnist = require('mnist');
const brain = require('brain.js');

// Load model weights
const modelJSON = JSON.parse(fs.readFileSync('mnist_model_weights.json', 'utf8'));

// Initialize the network with the saved weights
const net = new brain.NeuralNetwork();
net.fromJSON(modelJSON);

// Load MNIST test data
const set = mnist.set(0, 10);  // Use 10 samples from the test set
const testData = set.test;

// Predict and print results
testData.forEach((data, index) => {
    const output = net.run(data.input);
    const predictedLabel = output.indexOf(Math.max(...output));
    const actualLabel = data.output.indexOf(1);

    console.log(`Index ${index} - Actual Label: ${actualLabel}, Predicted Label: ${predictedLabel}`);
});
