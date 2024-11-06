const fs = require('fs');
const mnist = require('mnist');
const brain = require('brain.js');

// Load MNIST training data
const set = mnist.set(60000, 0);  // Use all 60,000 training images, no test images
const trainingData = set.training.map(data => ({
    input: data.input,
    output: Array(10).fill(0).map((_, i) => (i === data.output.indexOf(1) ? 1 : 0))  // One-hot encoded output
}));

// Set up the neural network
const net = new brain.NeuralNetwork({
    hiddenLayers: [128, 64]  // Configured to match your initial setup
});

// Train the network
console.log('Training the network...');
net.train(trainingData, {
    iterations: 20,           // You can adjust this for better accuracy
    errorThresh: 0.005,       // Stop when the error is sufficiently low
    log: true,
    logPeriod: 1
});

// Save the model weights
const modelJSON = net.toJSON();
fs.writeFileSync('mnist_model_weights.json', JSON.stringify(modelJSON));
console.log('Model weights saved to mnist_model_weights.json');
