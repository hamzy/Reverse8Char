// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Simple Neuron representing a single character's weight/state
type Neuron struct {
	Weights []float64
	Bias    float64
}

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidDerivative uses the output of the sigmoid function to calculate the gradient
func sigmoidDerivative(sigmoidOutput float64) float64 {
    return sigmoidOutput * (1.0 - sigmoidOutput)
}

// Process mimics forward propagation for a single step
func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return sigmoid(sum + n.Bias)
}

func (n *Neuron) Train(inputs []float64, target float64, learningRate float64) {
    // 1. Forward Pass: Get current prediction
    prediction := n.Process(inputs)

    // 2. Calculate Error: How far off were we?
    // Using simple difference; for multiple layers, you'd use Mean Squared Error derivative.
    errorDelta := target - prediction

    // 3. Calculate Gradient: Multiply error by the slope of the activation function
    gradient := errorDelta * sigmoidDerivative(prediction)

    // 4. Update Weights and Bias
    for i := range n.Weights {
        n.Weights[i] += inputs[i] * gradient * learningRate
    }
    n.Bias += gradient * learningRate
}

func stringToFloats(s string) []float64 {
	f := make([]float64, len(s))
	for i, r := range s {
		f[i] = float64(r) / 255.0
	}
	return f
}

func printResults(network []*Neuron, inputs []float64) {
	resultRunes := make([]rune, 8)

	for i := 0; i < 8; i++ {
		// 1. Get the raw prediction from the neuron (0.0 to 1.0)
		rawOutput := network[i].Process(inputs)

		// 2. Denormalize: convert 0.0-1.0 back to a byte value (0-255)
		// We add 0.5 before casting to int to ensure proper rounding.
		charValue := int(rawOutput*255.0 + 0.5)

		// 3. Store as a rune for character representation
		resultRunes[i] = rune(charValue)
	}

	fmt.Printf("Reversed Output: %s\n", string(resultRunes))
}

func main() {
	inputStr := "abcdefgh"
	targetStr := "hgfedcba"
	learningRate := 0.5
	epochs := 10000

	// Initialize 8 neurons
	network := make([]*Neuron, 8)
	for i := range network {
		network[i] = &Neuron{Weights: make([]float64, 8), Bias: rand.Float64()}
	}

	inputs := stringToFloats(inputStr)
	var wg sync.WaitGroup

	fmt.Println("Training in parallel...")

	// Launch a goroutine for each output position
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(pos int) {
			defer wg.Done()
			target := float64(targetStr[pos]) / 255.0

			// Each goroutine trains its specific neuron over all epochs
			for e := 0; e < epochs; e++ {
				network[pos].Train(inputs, target, learningRate)
			}
		}(i)
	}

	wg.Wait() // Block until all neurons are trained
	fmt.Println("Training Complete!")

	// Test the results
	printResults(network, inputs)
}
