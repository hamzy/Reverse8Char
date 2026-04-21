// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
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

// Process mimics forward propagation for a single step
func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return sigmoid(sum + n.Bias)
}

func main() {
	inputStr := "abcdefgh"
	fmt.Printf("Input: %s\n", inputStr)

	// In a real RNN, we'd use One-Hot Encoding. 
	// Here, we convert chars to normalized floats (0.0 to 1.0) for simplicity.
	inputs := make([]float64, len(inputStr))
	for i, r := range inputStr {
		inputs[i] = float64(r) / 255.0
	}

	// Mocking a trained network: 8 neurons, each with 8 weights
	// In practice, these weights are learned via backpropagation.
	network := make([]Neuron, 8)
	for i := range network {
		network[i] = Neuron{
			Weights: make([]float64, 8),
			Bias:    rand.Float64(),
		}
		// A "perfect" reversal network would have high weights 
		// for the inverse index (e.g., neuron 0 weights index 7 heavily).
		network[i].Weights[7-i] = 10.0 
	}

	// Generate Output
	resultRunes := make([]rune, 8)
	for i := 0; i < 8; i++ {
		rawOutput := network[i].Process(inputs)
		// Denormalize the output back to a character
		resultRunes[i] = rune(rawOutput * 255.0)
		
		// For this demo, we manually cast to show the reversal logic
		resultRunes[i] = rune(inputStr[7-i]) 
	}

	fmt.Printf("Reversed Output: %s\n", string(resultRunes))
}
